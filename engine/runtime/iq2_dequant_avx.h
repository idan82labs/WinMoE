#pragma once
/*
 * AVX2/AVX512 optimized IQ2_XXS dequant + dot product
 *
 * Autoresearch target: optimize this kernel until gate_proj < 0.6 ms
 * Current scalar baseline: 4.19 ms
 * Target: 0.5 ms (8.4x speedup)
 */

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

/*
 * AVX2 optimized: process 8 floats at a time using _mm256
 * The inner loop processes 8 grid values × 8 input values
 */
static inline float iq2xxs_dot_block_avx2(const block_iq2_xxs* block, const float* y) {
    float d = fp16_to_fp32(block->d);
    const uint16_t* q2 = block->qs;

    uint32_t aux32[2];
    const uint8_t* aux8 = (const uint8_t*)aux32;

    __m256 acc = _mm256_setzero_ps();

    for (int ib32 = 0; ib32 < QK_K/32; ib32++) {
        memcpy(aux32, q2, 8);
        q2 += 4;

        float ls = (float)(2 * (aux32[1] >> 28) + 1);
        __m256 vls = _mm256_set1_ps(ls);

        for (int l = 0; l < 4; l++) {
            const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
            uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7*l)) & 127];

            /* Load 8 grid values and convert to float */
            __m256 vgrid = _mm256_set_ps(
                (float)grid[7], (float)grid[6], (float)grid[5], (float)grid[4],
                (float)grid[3], (float)grid[2], (float)grid[1], (float)grid[0]
            );

            /* Apply signs */
            __m256 vsigns = _mm256_set_ps(
                (signs & 128) ? -1.0f : 1.0f,
                (signs & 64)  ? -1.0f : 1.0f,
                (signs & 32)  ? -1.0f : 1.0f,
                (signs & 16)  ? -1.0f : 1.0f,
                (signs & 8)   ? -1.0f : 1.0f,
                (signs & 4)   ? -1.0f : 1.0f,
                (signs & 2)   ? -1.0f : 1.0f,
                (signs & 1)   ? -1.0f : 1.0f
            );

            vgrid = _mm256_mul_ps(vgrid, vsigns);
            vgrid = _mm256_mul_ps(vgrid, vls);

            /* Load 8 input values */
            __m256 vy = _mm256_loadu_ps(&y[ib32 * 32 + l * 8]);

            /* FMA: acc += grid * y */
            acc = _mm256_fmadd_ps(vgrid, vy, acc);
        }
    }

    /* Horizontal sum */
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    float result;
    _mm_store_ss(&result, sum4);

    return d * result * 0.125f;
}

/*
 * AVX2 matrix-vector multiply
 */
static inline void iq2xxs_matvec_avx2(
    float* y, const void* W, const float* x,
    int out_dim, int in_dim
) {
    int blocks_per_row = in_dim / QK_K;
    const block_iq2_xxs* blocks = (const block_iq2_xxs*)W;

    for (int row = 0; row < out_dim; row++) {
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            sum += iq2xxs_dot_block_avx2(
                &blocks[row * blocks_per_row + b],
                &x[b * QK_K]
            );
        }
        y[row] = sum;
    }
}
