#pragma once
/*
 * Q5_K Dequantization — AVX-512 + Integer Accumulation
 *
 * Two paths:
 * 1. Float path (q5k_dot_block): AVX-512, used when no Q8_K activations available
 * 2. Integer path (q5k_dot_q8k): vpmaddubsw with pre-quantized Q8_K activations
 */

#include <stdint.h>
#include <string.h>
#include <malloc.h>
#include <immintrin.h>
#include "q8k_quant.h"

#define Q5K_BLOCK_SIZE 176
#define Q5K_QK 256

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
} block_q5_K;

/* AVX-512 float path (fallback) */
static inline float q5k_dot_block(const block_q5_K* block, const float* y) {
    float d = fp16_to_fp32(block->d);
    float dmin = fp16_to_fp32(block->dmin);
    uint8_t scales[8], mins[8];
    decode_q4k_scales(block->scales, scales, mins);
    const uint8_t* qs = block->qs;
    const uint8_t* qh = block->qh;
    __m512 acc = _mm512_setzero_ps();

    for (int sub = 0; sub < 8; sub++) {
        __m512 vsc = _mm512_set1_ps(d * (float)scales[sub]);
        __m512 vmn = _mm512_set1_ps(dmin * (float)mins[sub]);
        const float* yp = y + sub * 32;
        uint32_t hb;
        memcpy(&hb, qh + sub * 4, 4);

        /* First 16 weights */
        __m128i raw = _mm_loadl_epi64((const __m128i*)(qs + sub * 16));
        __m128i lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
        __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
        __m128i interleaved = _mm_unpacklo_epi8(lo, hi_nib);
        __m512i q4 = _mm512_cvtepu8_epi32(interleaved);
        uint16_t hb16 = (uint16_t)(hb & 0xFFFF);
        __m512i vhb = _mm512_set_epi32(
            (hb16>>15)&1,(hb16>>14)&1,(hb16>>13)&1,(hb16>>12)&1,
            (hb16>>11)&1,(hb16>>10)&1,(hb16>>9)&1,(hb16>>8)&1,
            (hb16>>7)&1,(hb16>>6)&1,(hb16>>5)&1,(hb16>>4)&1,
            (hb16>>3)&1,(hb16>>2)&1,(hb16>>1)&1,hb16&1);
        __m512i q5 = _mm512_or_epi32(q4, _mm512_slli_epi32(vhb, 4));
        __m512 vw = _mm512_sub_ps(_mm512_mul_ps(vsc, _mm512_cvtepi32_ps(q5)), vmn);
        acc = _mm512_fmadd_ps(vw, _mm512_loadu_ps(yp), acc);

        /* Second 16 weights */
        raw = _mm_loadl_epi64((const __m128i*)(qs + sub * 16 + 8));
        lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
        hi_nib = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
        interleaved = _mm_unpacklo_epi8(lo, hi_nib);
        q4 = _mm512_cvtepu8_epi32(interleaved);
        hb16 = (uint16_t)(hb >> 16);
        vhb = _mm512_set_epi32(
            (hb16>>15)&1,(hb16>>14)&1,(hb16>>13)&1,(hb16>>12)&1,
            (hb16>>11)&1,(hb16>>10)&1,(hb16>>9)&1,(hb16>>8)&1,
            (hb16>>7)&1,(hb16>>6)&1,(hb16>>5)&1,(hb16>>4)&1,
            (hb16>>3)&1,(hb16>>2)&1,(hb16>>1)&1,hb16&1);
        q5 = _mm512_or_epi32(q4, _mm512_slli_epi32(vhb, 4));
        vw = _mm512_sub_ps(_mm512_mul_ps(vsc, _mm512_cvtepi32_ps(q5)), vmn);
        acc = _mm512_fmadd_ps(vw, _mm512_loadu_ps(yp + 16), acc);
    }
    return _mm512_reduce_add_ps(acc);
}

/* Integer accumulation path with Q8_K activations */
static inline float q5k_dot_q8k(const block_q5_K* w, const block_q8_K* a) {
    float d = fp16_to_fp32(w->d) * a->d;
    float dmin = fp16_to_fp32(w->dmin) * a->d;
    uint8_t scales[8], mins_arr[8];
    decode_q4k_scales(w->scales, scales, mins_arr);

    const uint8_t* qs = w->qs;
    const uint8_t* qh = w->qh;
    const int8_t* q8 = a->qs;

    /* Min contribution via bsums */
    int32_t min_sum = 0;
    int sub;
    for (sub = 0; sub < 8; sub++)
        min_sum += (int32_t)mins_arr[sub] * (a->bsums[sub*2] + a->bsums[sub*2+1]);

    /* Load all 256 high bits */
    __m256i hbits = _mm256_loadu_si256((const __m256i*)qh);
    __m256i hmask = _mm256_set1_epi8(1);

    __m256i sumi = _mm256_setzero_si256();
    __m256i m4 = _mm256_set1_epi8(0x0F);

    for (sub = 0; sub < 8; sub++) {
        __m256i vscale = _mm256_set1_epi16((int16_t)scales[sub]);

        /* Load 16 bytes of qs → 32 nibbles (low and high halves) */
        __m128i q5raw = _mm_loadu_si128((const __m128i*)(qs + sub * 16));
        __m256i q5bytes = _mm256_set_m128i(_mm_srli_epi16(q5raw, 4), q5raw);
        __m256i q4lo = _mm256_and_si256(q5bytes, m4);

        /* Extract high bits: AND hbits with current mask, shift to bit 4 */
        __m256i q5h = _mm256_and_si256(hbits, hmask);
        /* Shift high bit into position 4 of each byte */
        /* q5h has 1 or 0 in each byte position. We need it at bit 4. */
        __m256i q5h_shifted = _mm256_slli_epi16(q5h, 4);
        /* Combine: OR in the high bit */
        __m256i q5v = _mm256_or_si256(q4lo, q5h_shifted);

        /* Advance high-bit mask for next sub-block */
        hmask = _mm256_slli_epi16(hmask, 1);

        /* Load Q8 values */
        __m256i q8v = _mm256_loadu_si256((const __m256i*)(q8 + sub * 32));

        /* u8 * s8 → s16, then scale */
        __m256i p16 = _mm256_maddubs_epi16(q5v, q8v);
        __m256i p32 = _mm256_madd_epi16(vscale, p16);
        sumi = _mm256_add_epi32(sumi, p32);
    }

    /* Convert to float */
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

/* Matvec with automatic pre-quantization */
static inline void q5k_matvec(
    float* out, const void* W, const float* x,
    int out_dim, int in_dim
) {
    int blocks_per_row = in_dim / Q5K_QK;
    const block_q5_K* weights = (const block_q5_K*)W;

    /* Pre-quantize activation to Q8_K */
    int num_blocks = in_dim / Q8K_QK;
    block_q8_K* x_q8 = (block_q8_K*)_malloca(num_blocks * sizeof(block_q8_K));
    if (!x_q8) {
        /* Float fallback */
        int row;
        #pragma omp parallel for schedule(static) if(out_dim >= 128)
        for (row = 0; row < out_dim; row++) {
            float sum = 0.0f;
            const block_q5_K* rb = &weights[row * blocks_per_row];
            for (int b = 0; b < blocks_per_row; b++)
                sum += q5k_dot_block(&rb[b], &x[b * Q5K_QK]);
            out[row] = sum;
        }
        return;
    }
    quantize_row_q8_K(x_q8, x, in_dim);

    int row;
    #pragma omp parallel for schedule(static) if(out_dim >= 128)
    for (row = 0; row < out_dim; row++) {
        float sum = 0.0f;
        const block_q5_K* rb = &weights[row * blocks_per_row];
        for (int b = 0; b < blocks_per_row; b++)
            sum += q5k_dot_q8k(&rb[b], &x_q8[b]);
        out[row] = sum;
    }
    _freea(x_q8);
}

/* Q5_K matvec with externally pre-quantized Q8_K activations */
static inline void q5k_matvec_q8k(
    float* out, const void* W, const block_q8_K* x_q8,
    int out_dim, int in_dim
) {
    int blocks_per_row = in_dim / Q5K_QK;
    const block_q5_K* weights = (const block_q5_K*)W;
    int row;
    #pragma omp parallel for schedule(static) if(out_dim >= 128)
    for (row = 0; row < out_dim; row++) {
        float sum = 0.0f;
        const block_q5_K* rb = &weights[row * blocks_per_row];
        for (int b = 0; b < blocks_per_row; b++)
            sum += q5k_dot_q8k(&rb[b], &x_q8[b]);
        out[row] = sum;
    }
}
