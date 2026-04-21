/* q6k_parity.cpp — verify AVX2 q6k_dot_block_avx2 vs scalar q6k_dot_block.
 *
 * Self-contained: synthesizes representative activations and Q6_K blocks
 * internally so it doesn't need real LM head bytes on disk.
 *
 * The key claim: q6k_dot_block_avx2 == q6k_dot_block (within FP rounding)
 * for any (block, activation) pair. The kernel does not branch on values,
 * so synthetic data covers the full input space — there's no data-dependent
 * code path. We use multiple distributions (Gaussian small, Gaussian large,
 * Q6_K-realistic [-32..31]) to exercise rounding patterns.
 *
 * Pass: max_abs_diff < 1e-3 across N pairs, rmse < 1e-4.
 *
 * Build (in MSVC dev shell):
 *   cl /O2 /arch:AVX2 /openmp /fp:fast /EHsc q6k_parity.cpp /Fe:q6k_parity.exe
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

/* fp16_to_fp32 is needed by q6k_dequant.h — define before include */
static inline float fp16_to_fp32(uint16_t h) {
    int sign = (h >> 15) & 1;
    int exp  = (h >> 10) & 0x1F;
    int mant = h & 0x3FF;
    if (exp == 0) return (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * (1.0f / 16384.0f);
    if (exp == 31) return sign ? -INFINITY : INFINITY;
    return (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, (float)(exp - 15));
}

#include "../q6k_dequant.h"

static uint16_t fp32_to_fp16(float v) {
    /* Quick & dirty fp32->fp16 (not bit-exact for subnormals/inf, fine for our random values) */
    uint32_t u; memcpy(&u, &v, 4);
    int sign = (u >> 31) & 1;
    int exp  = ((u >> 23) & 0xFF) - 127;
    int mant = u & 0x7FFFFF;
    if (exp >= 16) return (sign << 15) | 0x7C00;  // inf
    if (exp <= -15) return (sign << 15);          // zero
    return (uint16_t)((sign << 15) | ((exp + 15) << 10) | (mant >> 13));
}

static void gen_random_block(block_q6_K* blk, std::mt19937& rng, float scale_target) {
    // Random ql nibbles
    std::uniform_int_distribution<int> u8(0, 255);
    for (int i = 0; i < 128; ++i) blk->ql[i] = (uint8_t)u8(rng);
    for (int i = 0; i < 64;  ++i) blk->qh[i] = (uint8_t)u8(rng);
    // scales: int8, typically a few units; pick from [-30, 30]
    std::uniform_int_distribution<int> s8(-30, 30);
    for (int i = 0; i < 16; ++i) blk->scales[i] = (int8_t)s8(rng);
    // d: a typical FP16 scale; use scale_target
    blk->d = fp32_to_fp16(scale_target);
}

int main(int argc, char** argv) {
    int n_blocks = 1000;            // number of (block, activation) test pairs
    int n_act_distributions = 3;    // gaussian-small, gaussian-large, uniform-bounded
    unsigned long seed = 0xC0FFEE;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--n") && i+1 < argc) n_blocks = atoi(argv[++i]);
        if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = strtoul(argv[++i], nullptr, 10);
    }

    std::mt19937 rng(seed);

    double max_abs = 0.0, sum_sq = 0.0, max_rel = 0.0;
    int worst_dist = -1, worst_seed_idx = -1;
    double worst_scalar = 0, worst_avx = 0;
    long long n_pairs = 0;

    const char* dist_names[] = {"gauss_small_01", "gauss_large_3", "uniform_pm10"};

    for (int dist = 0; dist < n_act_distributions; ++dist) {
        for (int idx = 0; idx < n_blocks; ++idx) {
            // Generate one Q6_K block
            block_q6_K blk;
            float d_target = 0.001f + 0.01f * std::uniform_real_distribution<float>(0,1)(rng);
            gen_random_block(&blk, rng, d_target);

            // Generate activation vector of 256 floats per chosen distribution
            float act[Q6K_QK];
            if (dist == 0) {
                std::normal_distribution<float> gauss(0.0f, 0.1f);
                for (int i = 0; i < Q6K_QK; ++i) act[i] = gauss(rng);
            } else if (dist == 1) {
                std::normal_distribution<float> gauss(0.0f, 3.0f);
                for (int i = 0; i < Q6K_QK; ++i) act[i] = gauss(rng);
            } else {
                std::uniform_real_distribution<float> uni(-10.0f, 10.0f);
                for (int i = 0; i < Q6K_QK; ++i) act[i] = uni(rng);
            }

            float scalar = q6k_dot_block(&blk, act);
            float avx    = q6k_dot_block_avx2(&blk, act);
            double diff  = (double)avx - (double)scalar;
            double abs_diff = fabs(diff);
            double rel_diff = abs_diff / (fabs((double)scalar) + 1e-9);
            sum_sq += diff * diff;
            n_pairs++;
            if (abs_diff > max_abs) {
                max_abs = abs_diff;
                worst_dist = dist; worst_seed_idx = idx;
                worst_scalar = scalar; worst_avx = avx;
            }
            if (rel_diff > max_rel) max_rel = rel_diff;
        }
    }

    double rmse = sqrt(sum_sq / (double)n_pairs);

    printf("=== q6k AVX2 vs scalar parity ===\n");
    printf("pairs tested:    %lld (%d distributions × %d each)\n",
           n_pairs, n_act_distributions, n_blocks);
    printf("max_abs_diff:    %.6e\n", max_abs);
    printf("rmse:            %.6e\n", rmse);
    printf("max_rel_diff:    %.6e\n", max_rel);
    printf("worst dist:      %s (idx %d)\n",
           worst_dist >= 0 ? dist_names[worst_dist] : "n/a", worst_seed_idx);
    printf("worst scalar:    %.6f\n", worst_scalar);
    printf("worst avx:       %.6f\n", worst_avx);

    bool pass_abs  = max_abs < 1e-3;
    bool pass_rmse = rmse < 1e-4;
    printf("\n");
    printf("max_abs_diff < 1e-3: %s\n", pass_abs ? "PASS" : "FAIL");
    printf("rmse < 1e-4:         %s\n", pass_rmse ? "PASS" : "FAIL");
    printf("\nRESULT: %s\n", (pass_abs && pass_rmse) ? "PASS" : "FAIL");
    return (pass_abs && pass_rmse) ? 0 : 1;
}
