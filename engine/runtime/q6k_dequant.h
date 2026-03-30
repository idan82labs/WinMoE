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

#define Q6K_BLOCK_SIZE 210
#define Q6K_QK 256

typedef struct {
    uint8_t ql[128];    /* lower 4 bits */
    uint8_t qh[64];     /* upper 2 bits */
    int8_t  scales[16]; /* per-group scales */
    uint16_t d;         /* FP16 super-block scale */
} block_q6_K;

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
            int is = n / 16;  /* scale base index */
            /* 4 weights per (n, l) combination */
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

static inline void q6k_matvec(
    float* out, const void* W, const float* x,
    int out_dim, int in_dim
) {
    int blocks_per_row = in_dim / Q6K_QK;
    const block_q6_K* blocks = (const block_q6_K*)W;

    int row;
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
}
