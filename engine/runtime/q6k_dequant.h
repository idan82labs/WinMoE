#pragma once
/*
 * Q6_K Dequantization — 6-bit quantization (6.5625 bpw)
 *
 * Block: 256 weights in 210 bytes
 *   ql[128]: lower 4 bits (2 per byte)
 *   qh[64]:  upper 2 bits (4 per byte)
 *   scales[16]: int8 scales per 16-weight group
 *   d: FP16 super-block scale
 *
 * Dequant: weight = d * scale * ((ql & 0xF) | ((qh & 3) << 4) - 32)
 */

#include <stdint.h>
#include <immintrin.h>

#define Q6K_BLOCK_SIZE 210
#define Q6K_QK 256

typedef struct {
    uint8_t ql[128];    /* lower 4 bits */
    uint8_t qh[64];     /* upper 2 bits */
    int8_t  scales[16]; /* per-group scales */
    uint16_t d;         /* FP16 super-block scale */
} block_q6_K;

/* AVX2-vectorized Q6_K dot product.
 * Processes 8 l-values at a time (across the 32-wide inner loop).
 * ~2-3x faster than scalar on Tiger Lake-H (i7-11800H, AVX2 but no AVX-512). */
static inline float q6k_dot_block_avx2(const block_q6_K* block, const float* y) {
    float d = fp16_to_fp32(block->d);
    __m256 acc = _mm256_setzero_ps();
    const __m128i mask_4 = _mm_set1_epi8(0x0F);
    const __m128i mask_2 = _mm_set1_epi8(0x03);
    const __m256i v32 = _mm256_set1_epi32(32);

    for (int n = 0; n < 256; n += 128) {
        /* Q6_K layout uses 8 scales per 128-weight half, applied at 16-weight granularity.
         * l<16 uses scales[(n/16)+{0,2,4,6}], l>=16 uses scales[(n/16)+{1,3,5,7}].
         * Earlier code used n/16 for the whole l-range, dropping half the scales and
         * compressing high-magnitude rows. We handle the two halves separately. */
        int is_lo = (n / 16);
        int is_hi = (n / 16) + 1;
        __m256 vs1_lo = _mm256_set1_ps(d * (float)block->scales[is_lo+0]);
        __m256 vs2_lo = _mm256_set1_ps(d * (float)block->scales[is_lo+2]);
        __m256 vs3_lo = _mm256_set1_ps(d * (float)block->scales[is_lo+4]);
        __m256 vs4_lo = _mm256_set1_ps(d * (float)block->scales[is_lo+6]);
        __m256 vs1_hi = _mm256_set1_ps(d * (float)block->scales[is_hi+0]);
        __m256 vs2_hi = _mm256_set1_ps(d * (float)block->scales[is_hi+2]);
        __m256 vs3_hi = _mm256_set1_ps(d * (float)block->scales[is_hi+4]);
        __m256 vs4_hi = _mm256_set1_ps(d * (float)block->scales[is_hi+6]);

        for (int l = 0; l < 32; l += 8) {
            int hi = (l >= 16);
            __m256 vs1 = hi ? vs1_hi : vs1_lo;
            __m256 vs2 = hi ? vs2_hi : vs2_lo;
            __m256 vs3 = hi ? vs3_hi : vs3_lo;
            __m256 vs4 = hi ? vs4_hi : vs4_lo;
            /* Load 8 qh bytes (each byte holds 4 × 2-bit values for q1..q4) */
            __m128i qh = _mm_loadl_epi64((const __m128i*)(block->qh + l + n/4));
            __m128i qh_q1 = _mm_and_si128(qh, mask_2);
            __m128i qh_q2 = _mm_and_si128(_mm_srli_epi16(qh, 2), mask_2);
            __m128i qh_q3 = _mm_and_si128(_mm_srli_epi16(qh, 4), mask_2);
            __m128i qh_q4 = _mm_and_si128(_mm_srli_epi16(qh, 6), mask_2);

            /* Load 8 ql bytes for q1 (low nibble) + q3 (high nibble) */
            __m128i ql_q1q3 = _mm_loadl_epi64((const __m128i*)(block->ql + l + 0  + n/2));
            __m128i ql_q1 = _mm_and_si128(ql_q1q3, mask_4);
            __m128i ql_q3 = _mm_and_si128(_mm_srli_epi16(ql_q1q3, 4), mask_4);

            /* Load 8 ql bytes for q2 + q4 */
            __m128i ql_q2q4 = _mm_loadl_epi64((const __m128i*)(block->ql + l + 32 + n/2));
            __m128i ql_q2 = _mm_and_si128(ql_q2q4, mask_4);
            __m128i ql_q4 = _mm_and_si128(_mm_srli_epi16(ql_q2q4, 4), mask_4);

            /* Combine: 6-bit value = ql_low | (qh << 4) */
            __m128i q1 = _mm_or_si128(ql_q1, _mm_slli_epi16(qh_q1, 4));
            __m128i q2 = _mm_or_si128(ql_q2, _mm_slli_epi16(qh_q2, 4));
            __m128i q3 = _mm_or_si128(ql_q3, _mm_slli_epi16(qh_q3, 4));
            __m128i q4 = _mm_or_si128(ql_q4, _mm_slli_epi16(qh_q4, 4));

            /* Expand 8×u8 -> 8×i32, subtract 32 (Q6_K bias), convert to float */
            __m256 q1f = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q1), v32));
            __m256 q2f = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q2), v32));
            __m256 q3f = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q3), v32));
            __m256 q4f = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q4), v32));

            /* Load 8 activations for each of 4 positions (l+0, l+32, l+64, l+96 within n-block) */
            __m256 y1 = _mm256_loadu_ps(y + l + 0  + n);
            __m256 y2 = _mm256_loadu_ps(y + l + 32 + n);
            __m256 y3 = _mm256_loadu_ps(y + l + 64 + n);
            __m256 y4 = _mm256_loadu_ps(y + l + 96 + n);

            /* acc += (scale_i * q_i) * y_i for i=1..4 */
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vs1, q1f), y1, acc);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vs2, q2f), y2, acc);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vs3, q3f), y3, acc);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vs4, q4f), y4, acc);
        }
    }
    /* Horizontal sum of 8-wide acc */
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    float sum;
    _mm_store_ss(&sum, s4);
    return sum;
}

static inline float q6k_dot_block(const block_q6_K* block, const float* y) {
    float d = fp16_to_fp32(block->d);
    float sum = 0.0f;

    /* GGML Q6_K layout: 256 weights in 2 blocks of 128.
     * Each 128-weight block has 4 groups of 32 weights.
     * ql: 32 low nibbles per group, packed as [0..31][32..63] then high nibbles
     * qh: 32 bytes, each holding 4×2-bit values for the 4 groups
     * Scales: 16 int8 values, 2 per 128-block (8 per block, but interleaved) */
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; l++) {
            /* scale index: (n/16) picks the 128-half (0 or 8), (l/16) picks
             * which of the two 16-weight sub-blocks within this half (0 or 1).
             * Earlier `is = n/16` ignored l/16, collapsing two distinct scales
             * into one and selectively compressing high-magnitude rows. */
            int is = (n / 16) + (l / 16);
            int q1 = (block->ql[l + 0  + n/2] & 0xF) | (((block->qh[l + n/4] >> 0) & 3) << 4);
            int q2 = (block->ql[l + 32 + n/2] & 0xF) | (((block->qh[l + n/4] >> 2) & 3) << 4);
            int q3 = (block->ql[l + 0  + n/2] >> 4)  | (((block->qh[l + n/4] >> 4) & 3) << 4);
            int q4 = (block->ql[l + 32 + n/2] >> 4)  | (((block->qh[l + n/4] >> 6) & 3) << 4);

            sum += d * (float)block->scales[is+0] * (float)(q1 - 32) * y[l + 0  + n];
            sum += d * (float)block->scales[is+2] * (float)(q2 - 32) * y[l + 32 + n];
            sum += d * (float)block->scales[is+4] * (float)(q3 - 32) * y[l + 64 + n];
            sum += d * (float)block->scales[is+6] * (float)(q4 - 32) * y[l + 96 + n];
        }
    }

    return sum;
}

extern int g_q6k_use_scalar;  /* set by main from WINMOE_LM_SCALAR env */

static inline void q6k_matvec(
    float* out, const void* W, const float* x,
    int out_dim, int in_dim
) {
    int blocks_per_row = in_dim / Q6K_QK;
    const block_q6_K* blocks = (const block_q6_K*)W;

    int row;
    if (g_q6k_use_scalar) {
        #pragma omp parallel for schedule(static) if(out_dim >= 128)
        for (row = 0; row < out_dim; row++) {
            float sum = 0.0f;
            const block_q6_K* row_blocks = &blocks[row * blocks_per_row];
            int b;
            for (b = 0; b < blocks_per_row; b++) {
                sum += q6k_dot_block(&row_blocks[b], &x[b * Q6K_QK]);
            }
            out[row] = sum;
        }
    } else {
        #pragma omp parallel for schedule(static) if(out_dim >= 128)
        for (row = 0; row < out_dim; row++) {
            float sum = 0.0f;
            const block_q6_K* row_blocks = &blocks[row * blocks_per_row];
            int b;
            for (b = 0; b < blocks_per_row; b++) {
                sum += q6k_dot_block_avx2(&row_blocks[b], &x[b * Q6K_QK]);
            }
            out[row] = sum;
        }
    }
}
