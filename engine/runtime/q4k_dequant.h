#pragma once
/*
 * Q4_K_M Dequantization — Standard GGML format
 *
 * Much simpler than IQ2_XXS: no lookup tables, just linear scale+min.
 *
 * Block structure (144 bytes per 256 weights):
 *   d:      FP16 super-block scale
 *   dmin:   FP16 super-block min
 *   scales: uint8[12] — packed 6-bit scales and mins for 8 sub-blocks
 *   qs:     uint8[128] — packed 4-bit quantized values (2 per byte)
 *
 * Dequant: weight = d * scale * (q4_value) - dmin * min_value
 *
 * For Q4_K, each 256-weight block has 8 sub-blocks of 32 weights.
 * Each sub-block has its own 6-bit scale and 6-bit min.
 */

#include <stdint.h>
#include <string.h>
#include <malloc.h>
#include <immintrin.h>
#include "q8k_quant.h"

#define Q4K_BLOCK_SIZE 144
#define Q4K_QK 256

typedef struct {
    uint16_t d;         /* FP16 super-block scale */
    uint16_t dmin;      /* FP16 super-block min */
    uint8_t scales[12]; /* packed 6-bit scales and mins */
    uint8_t qs[128];    /* packed 4-bit weights (256 values, 2 per byte) */
} block_q4_K;

/*
 * Decode 6-bit scales and mins from packed format.
 * 12 bytes encode 8 scales + 8 mins, each 6 bits.
 * Layout (from GGML):
 *   bytes 0-3:  low 4 bits of scales[0..7] (packed as nibbles)
 *   bytes 4-7:  low 4 bits of mins[0..7]
 *   bytes 8-11: high 2 bits of scales[0..7] and mins[0..7]
 */
static inline void decode_q4k_scales(const uint8_t* packed, uint8_t* scales, uint8_t* mins) {
    /* Low 4 bits of scales */
    scales[0] = packed[0] & 0x3F;
    scales[1] = packed[0] >> 6 | (packed[1] & 0x0F) << 2;
    scales[2] = packed[1] >> 4 | (packed[2] & 0x03) << 4;
    scales[3] = packed[2] >> 2;
    scales[4] = packed[3] & 0x3F;
    scales[5] = packed[3] >> 6 | (packed[4] & 0x0F) << 2;
    scales[6] = packed[4] >> 4 | (packed[5] & 0x03) << 4;
    scales[7] = packed[5] >> 2;

    /* Mins */
    mins[0] = packed[6] & 0x3F;
    mins[1] = packed[6] >> 6 | (packed[7] & 0x0F) << 2;
    mins[2] = packed[7] >> 4 | (packed[8] & 0x03) << 4;
    mins[3] = packed[8] >> 2;
    mins[4] = packed[9] & 0x3F;
    mins[5] = packed[9] >> 6 | (packed[10] & 0x0F) << 2;
    mins[6] = packed[10] >> 4 | (packed[11] & 0x03) << 4;
    mins[7] = packed[11] >> 2;
}

/*
 * AVX-512 / AVX2 fused dequant + dot product for one Q4_K block
 *
 * AVX-512 path: processes 16 weights per iteration using unpacklo nibble interleave
 * AVX2 fallback: processes 8 weights per iteration
 * Both eliminate the scalar _mm256_set_epi32 antipattern
 */
static inline float q4k_dot_block(const block_q4_K* block, const float* y) {
    float d = fp16_to_fp32(block->d);
    float dmin = fp16_to_fp32(block->dmin);

    uint8_t scales[8], mins[8];
    decode_q4k_scales(block->scales, scales, mins);

    const uint8_t* qs = block->qs;
    __m512 acc = _mm512_setzero_ps();

    for (int sub = 0; sub < 8; sub++) {
        __m512 vsc = _mm512_set1_ps(d * (float)scales[sub]);
        __m512 vmn = _mm512_set1_ps(dmin * (float)mins[sub]);
        const uint8_t* qp = qs + sub * 16;
        const float* yp = y + sub * 32;

        /* First 8 bytes → 16 nibbles → 16 weights */
        __m128i raw = _mm_loadl_epi64((const __m128i*)qp);
        __m128i lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
        __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
        __m128i interleaved = _mm_unpacklo_epi8(lo, hi);
        __m512i q = _mm512_cvtepu8_epi32(interleaved);
        __m512 vw = _mm512_sub_ps(_mm512_mul_ps(vsc, _mm512_cvtepi32_ps(q)), vmn);
        acc = _mm512_fmadd_ps(vw, _mm512_loadu_ps(yp), acc);

        /* Second 8 bytes → next 16 weights */
        raw = _mm_loadl_epi64((const __m128i*)(qp + 8));
        lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
        hi = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
        interleaved = _mm_unpacklo_epi8(lo, hi);
        q = _mm512_cvtepu8_epi32(interleaved);
        vw = _mm512_sub_ps(_mm512_mul_ps(vsc, _mm512_cvtepi32_ps(q)), vmn);
        acc = _mm512_fmadd_ps(vw, _mm512_loadu_ps(yp + 16), acc);
    }

    return _mm512_reduce_add_ps(acc);
}

/*
 * Q4_K dot product with Q8_K pre-quantized activations (integer accumulation)
 * Uses vpmaddubsw for u8*s8 → s16 multiply, vpmaddwd for scale, stays in int32
 * Only converts to float ONCE per 256-weight block
 */
static inline float q4k_dot_q8k(const block_q4_K* w, const block_q8_K* a) {
    float d = fp16_to_fp32(w->d) * a->d;
    float dmin = fp16_to_fp32(w->dmin) * a->d;

    uint8_t scales[8], mins_arr[8];
    decode_q4k_scales(w->scales, scales, mins_arr);

    const uint8_t* qs = w->qs;
    const int8_t* q8 = a->qs;

    /* Compute min contribution using bsums */
    int32_t min_sum = 0;
    int sub;
    for (sub = 0; sub < 8; sub++) {
        min_sum += (int32_t)mins_arr[sub] * (a->bsums[sub*2] + a->bsums[sub*2+1]);
    }

    /* AVX2 integer dot product — maddubs + madd (optimal for per-sub-block scales) */
    __m256i sumi = _mm256_setzero_si256();
    __m256i m4 = _mm256_set1_epi8(0x0F);

    for (sub = 0; sub < 8; sub++) {
        __m256i vscale = _mm256_set1_epi16((int16_t)scales[sub]);

        __m128i q4raw = _mm_loadu_si128((const __m128i*)(qs + sub * 16));
        __m256i q4bytes = _mm256_set_m128i(_mm_srli_epi16(q4raw, 4), q4raw);
        q4bytes = _mm256_and_si256(q4bytes, m4);

        __m256i q8v = _mm256_loadu_si256((const __m256i*)(q8 + sub * 32));

        __m256i p16 = _mm256_maddubs_epi16(q4bytes, q8v);
        __m256i p32 = _mm256_madd_epi16(vscale, p16);
        sumi = _mm256_add_epi32(sumi, p32);
    }

    /* Convert accumulated int32 to float */
    __m256 vd = _mm256_set1_ps(d);
    __m256 vsum = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(sumi));
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    float result;
    _mm_store_ss(&result, s4);
    return result - dmin * (float)min_sum;
}

/*
 * Q4_K matrix-vector multiply
 * Automatically pre-quantizes activation to Q8_K for integer dot products
 */
static inline void q4k_matvec(
    float* out, const void* W, const float* x,
    int out_dim, int in_dim
) {
    int blocks_per_row = in_dim / Q4K_QK;
    const block_q4_K* weights = (const block_q4_K*)W;

    /* Pre-quantize activation vector to Q8_K (once, reused for all rows) */
    int num_blocks = in_dim / Q8K_QK;
    block_q8_K* x_q8 = (block_q8_K*)_malloca(num_blocks * sizeof(block_q8_K));
    if (!x_q8) {
        /* Fallback to float path */
        int row;
        #pragma omp parallel for schedule(static) if(out_dim >= 128)
        for (row = 0; row < out_dim; row++) {
            float sum = 0.0f;
            const block_q4_K* row_blocks = &weights[row * blocks_per_row];
            for (int b = 0; b < blocks_per_row; b++)
                sum += q4k_dot_block(&row_blocks[b], &x[b * Q4K_QK]);
            out[row] = sum;
        }
        return;
    }
    quantize_row_q8_K(x_q8, x, in_dim);

    int row;
    #pragma omp parallel for schedule(static) if(out_dim >= 128)
    for (row = 0; row < out_dim; row++) {
        float sum = 0.0f;
        const block_q4_K* row_blocks = &weights[row * blocks_per_row];
        for (int b = 0; b < blocks_per_row; b++) {
            sum += q4k_dot_q8k(&row_blocks[b], &x_q8[b]);
        }
        out[row] = sum;
    }
    _freea(x_q8);
}

/* Q4_K matvec with externally pre-quantized Q8_K activations
   Software prefetching for next row's weight data */
static inline void q4k_matvec_q8k(
    float* out, const void* W, const block_q8_K* x_q8,
    int out_dim, int in_dim
) {
    int blocks_per_row = in_dim / Q4K_QK;
    const block_q4_K* weights = (const block_q4_K*)W;
    int row;
    #pragma omp parallel for schedule(static) if(out_dim >= 128)
    for (row = 0; row < out_dim; row++) {
        /* Prefetch next row's first blocks while computing current */
        if (row + 1 < out_dim) {
            _mm_prefetch((const char*)&weights[(row+1) * blocks_per_row], _MM_HINT_T0);
            _mm_prefetch((const char*)&weights[(row+1) * blocks_per_row] + 64, _MM_HINT_T0);
        }
        float sum = 0.0f;
        const block_q4_K* rb = &weights[row * blocks_per_row];
        for (int b = 0; b < blocks_per_row; b++)
            sum += q4k_dot_q8k(&rb[b], &x_q8[b]);
        out[row] = sum;
    }
}
