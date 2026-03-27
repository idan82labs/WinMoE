#pragma once
/*
 * Q8_K Activation Quantization
 *
 * Quantize a float activation vector to Q8_K format for integer dot products.
 * Called once per matvec, then reused across all output rows.
 *
 * Q8_K block: 256 values
 *   - d: float32 scale (not fp16 — full precision for activations)
 *   - qs[256]: int8 quantized values
 *   - bsums[16]: int16 sums of groups of 16 values (for Q4_K min optimization)
 *
 * Quantize: qs[i] = round(x[i] / d), where d = max(|x|) / 127
 */

#include <stdint.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>

#define Q8K_QK 256

typedef struct {
    float d;              /* scale */
    int8_t qs[Q8K_QK];   /* quantized values */
    int16_t bsums[16];   /* sums per 16-value group (for min optimization) */
} block_q8_K;

/*
 * Quantize one block of 256 floats to Q8_K
 */
static inline void quantize_block_q8_K(block_q8_K* out, const float* x) {
    /* Find max absolute value */
    __m256 vmax = _mm256_setzero_ps();
    int i;
    for (i = 0; i < Q8K_QK; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        /* abs via AND with sign-bit mask */
        __m256 vabs = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vx);
        vmax = _mm256_max_ps(vmax, vabs);
    }
    /* Horizontal max */
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 m4 = _mm_max_ps(lo, hi);
    m4 = _mm_max_ps(m4, _mm_shuffle_ps(m4, m4, _MM_SHUFFLE(2,3,0,1)));
    m4 = _mm_max_ps(m4, _mm_shuffle_ps(m4, m4, _MM_SHUFFLE(1,0,3,2)));
    float amax;
    _mm_store_ss(&amax, m4);

    if (amax < 1e-10f) {
        out->d = 0.0f;
        memset(out->qs, 0, Q8K_QK);
        memset(out->bsums, 0, sizeof(out->bsums));
        return;
    }

    float d = amax / 127.0f;
    float id = 1.0f / d;
    out->d = d;

    /* Quantize and compute block sums */
    __m256 vid = _mm256_set1_ps(id);
    for (i = 0; i < Q8K_QK; i += 32) {
        int16_t bsum = 0;
        /* Process 32 values (matches Q4_K sub-block alignment) */
        for (int j = 0; j < 32; j += 8) {
            __m256 vx = _mm256_loadu_ps(x + i + j);
            __m256 vq = _mm256_mul_ps(vx, vid);
            /* Round to nearest */
            __m256i vi = _mm256_cvtps_epi32(vq); /* rounds to nearest by default */
            /* Clamp to [-127, 127] */
            vi = _mm256_max_epi32(vi, _mm256_set1_epi32(-127));
            vi = _mm256_min_epi32(vi, _mm256_set1_epi32(127));
            /* Pack int32 to int8 */
            __m128i lo32 = _mm256_castsi256_si128(vi);
            __m128i hi32 = _mm256_extracti128_si256(vi, 1);
            __m128i lo16 = _mm_packs_epi32(lo32, hi32); /* 8 x int16 */
            __m128i i8 = _mm_packs_epi16(lo16, lo16);   /* 8 x int8 (lower 8 bytes) */
            _mm_storel_epi64((__m128i*)(out->qs + i + j), i8);
            /* Accumulate block sum */
            for (int k = 0; k < 8; k++) bsum += out->qs[i + j + k];
        }
        out->bsums[i / 16] = bsum; /* Note: bsums are per 16 values, we have 32 here */
    }

    /* Fix bsums: should be per 16 values */
    for (i = 0; i < 16; i++) {
        int16_t s = 0;
        for (int j = 0; j < 16; j++) s += out->qs[i * 16 + j];
        out->bsums[i] = s;
    }
}

/*
 * Quantize a full activation vector to Q8_K format
 * in_dim must be a multiple of Q8K_QK (256)
 */
static inline void quantize_row_q8_K(block_q8_K* out, const float* x, int n) {
    int blocks = n / Q8K_QK;
    for (int b = 0; b < blocks; b++) {
        quantize_block_q8_K(&out[b], x + b * Q8K_QK);
    }
}
