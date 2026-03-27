#pragma once
/*
 * Optimized IQ2_XXS matmul — focus on reducing overhead
 *
 * Key insight: the LUT lookups are the bottleneck, not arithmetic.
 * Strategy: minimize overhead, prefetch LUT, batch operations.
 */

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

/* Precomputed sign table: 256 entries × 8 floats = +1/-1 patterns */
static float sign_table[256][8];
static int sign_table_init = 0;

static void init_sign_table(void) {
    if (sign_table_init) return;
    int i, j;
    for (i = 0; i < 256; i++) {
        for (j = 0; j < 8; j++) {
            sign_table[i][j] = (i & (1 << j)) ? -1.0f : 1.0f;
        }
    }
    sign_table_init = 1;
}

/*
 * Optimized: precomputed sign table eliminates _mm256_set_epi32 overhead
 */
static inline float iq2xxs_dot_block_fast(const block_iq2_xxs* block, const float* y) {
    float d = fp16_to_fp32(block->d);
    if (d != d || d > 1e30f || d < -1e30f) d = 0.0f;
    const uint16_t* q2 = block->qs;

    uint32_t aux32[2];
    const uint8_t* aux8 = (const uint8_t*)aux32;

    float bsum = 0.0f;

    for (int ib32 = 0; ib32 < QK_K/32; ib32++) {
        memcpy(aux32, q2, 8);
        q2 += 4;

        float ls = (float)(2 * (aux32[1] >> 28) + 1);
        const float* yp = &y[ib32 * 32];

        /* AVX2: process each group of 8 with vector ops */
        __m256 vacc = _mm256_setzero_ps();

        for (int l = 0; l < 4; l++) {
            const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
            uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7*l)) & 127];

            /* Convert 8 grid values to float */
            __m128i vgrid_i8 = _mm_loadl_epi64((const __m128i*)grid);
            __m256i vgrid_i32 = _mm256_cvtepu8_epi32(vgrid_i8);
            __m256 vgrid = _mm256_cvtepi32_ps(vgrid_i32);

            /* Apply signs from precomputed table — single aligned load */
            __m256 vsign = _mm256_loadu_ps(sign_table[signs]);
            vgrid = _mm256_mul_ps(vgrid, vsign);

            /* FMA with input */
            __m256 vy = _mm256_loadu_ps(yp + l * 8);
            vacc = _mm256_fmadd_ps(vgrid, vy, vacc);
        }

        /* Horizontal sum */
        __m128 hi = _mm256_extractf128_ps(vacc, 1);
        __m128 lo = _mm256_castps256_ps128(vacc);
        __m128 sum4 = _mm_add_ps(lo, hi);
        sum4 = _mm_hadd_ps(sum4, sum4);
        sum4 = _mm_hadd_ps(sum4, sum4);
        float group_sum;
        _mm_store_ss(&group_sum, sum4);

        bsum += group_sum * ls;
    }

    return d * bsum * 0.125f;
}

/*
 * Multi-threaded matmul: use OpenMP to parallelize rows
 */
static inline void iq2xxs_matvec_fast(
    float* y, const void* W, const float* x,
    int out_dim, int in_dim
) {
    int blocks_per_row = in_dim / QK_K;
    const block_iq2_xxs* blocks = (const block_iq2_xxs*)W;

    init_sign_table();

    /* Parallelize across rows */
    int row;
    #pragma omp parallel for schedule(static)
    for (row = 0; row < out_dim; row++) {
        float sum = 0.0f;
        const block_iq2_xxs* row_blocks = &blocks[row * blocks_per_row];
        for (int b = 0; b < blocks_per_row; b++) {
            sum += iq2xxs_dot_block_fast(&row_blocks[b], &x[b * QK_K]);
        }
        y[row] = sum;
    }
}
