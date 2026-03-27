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

    for (int group = 0; group < 16; group++) {
        float sc = d * (float)block->scales[group];
        int base = group * 16;

        for (int j = 0; j < 16; j++) {
            int idx = base + j;
            /* Extract 6-bit value: 4 low bits from ql, 2 high bits from qh */
            int ql_byte = idx / 2;
            int ql_shift = (idx % 2) * 4;
            int low4 = (block->ql[ql_byte] >> ql_shift) & 0x0F;

            int qh_byte = idx / 4;
            int qh_shift = (idx % 4) * 2;
            int high2 = (block->qh[qh_byte] >> qh_shift) & 0x03;

            int q6 = low4 | (high2 << 4);
            float w = sc * (float)(q6 - 32);
            sum += w * y[idx];
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
