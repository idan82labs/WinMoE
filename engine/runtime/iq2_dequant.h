#pragma once
/*
 * IQ2_XXS Dequantization — Ported from llama.cpp ggml-common.h
 *
 * Format: 256-weight super-blocks
 *   - d: FP16 scale for the block
 *   - qs[32]: packed indices and sign/scale data (16-bit words)
 *
 * Each 32-weight sub-block contains:
 *   - 4 grid indices (8 bits each) → look up iq2xxs_grid[idx] for 8 values
 *   - 7 sign bits → look up ksigns_iq2xs[signs] for 8 sign flags
 *   - 4-bit sub-scale ls = 2*(bits>>28)+1
 *
 * Dequantized value = d * ls * grid_value * sign * 0.125
 */

#include <stdint.h>
#include <string.h>
#include <math.h>

// FP16 conversion
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        // Denormalized
        float val = ldexpf((float)mant, -24);
        return sign ? -val : val;
    }
    if (exp == 0x1f) {
        if (mant == 0) return sign ? -INFINITY : INFINITY;
        return NAN;
    }

    float val = ldexpf((float)(mant + 1024), (int)exp - 25);
    return sign ? -val : val;
}

// Lookup tables from llama.cpp (ggml-common.h)
static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

// iq2xxs_grid: 256 entries, each is 8 bytes (8 uint8 values)
// Must be defined before including this header.
// Include iq2xxs_grid_def.h before this file.

// Block structure
#define QK_K 256  // super-block size

typedef struct {
    uint16_t d;           // FP16 scale
    uint16_t qs[QK_K/8]; // 32 uint16 = 64 bytes of packed data
} block_iq2_xxs;
// Total: 2 + 64 = 66 bytes per 256 weights = 2.0625 bpw

/*
 * Fused dequant + dot product: compute dot(dequant(block), y) for one block
 * This is the critical inner kernel.
 *
 * block: pointer to IQ2_XXS block (66 bytes, 256 weights)
 * y: pointer to FP32 input vector segment (256 floats)
 * Returns: dot product (float)
 */
static inline float iq2xxs_dot_block(const block_iq2_xxs* block, const float* y) {
    float d = fp16_to_fp32(block->d);
    /* Clamp NaN/Inf from Unsloth Dynamic quant artifacts */
    if (d != d || d > 1e30f || d < -1e30f) d = 0.0f;
    const uint16_t* q2 = block->qs;

    uint32_t aux32[2];
    const uint8_t* aux8 = (const uint8_t*)aux32;

    float bsum = 0.0f;

    for (int ib32 = 0; ib32 < QK_K/32; ib32++) {
        memcpy(aux32, q2, 2 * sizeof(uint32_t));
        q2 += 4;

        uint32_t ls = 2 * (aux32[1] >> 28) + 1;
        float sumi = 0.0f;

        for (int l = 0; l < 4; l++) {
            const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
            uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7*l)) & 127];

            for (int j = 0; j < 8; j++) {
                float w = (float)grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                sumi += w * y[ib32 * 32 + l * 8 + j];
            }
        }
        bsum += sumi * (float)ls;
    }

    return d * bsum * 0.125f;
}

/*
 * Matrix-vector multiply: y = W @ x
 * W is (out_dim, in_dim) in IQ2_XXS format
 * x is (in_dim,) FP32
 * y is (out_dim,) FP32
 */
static inline void iq2xxs_matvec(
    float* y, const void* W, const float* x,
    int out_dim, int in_dim
) {
    int blocks_per_row = in_dim / QK_K;
    const block_iq2_xxs* blocks = (const block_iq2_xxs*)W;

    for (int row = 0; row < out_dim; row++) {
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            sum += iq2xxs_dot_block(
                &blocks[row * blocks_per_row + b],
                &x[b * QK_K]
            );
        }
        y[row] = sum;
    }
}
