/*
 * Test Q4_K_M dequantization and matmul
 * Uses synthetic data first, then real model data when available
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <windows.h>

/* Grid table needed by IQ2 header (included transitively) */
#include "iq2xxs_grid_def.h"
#include "iq2_dequant.h"
#include "q4k_dequant.h"

int main(void) {
    printf("=== Q4_K_M Dequant Test ===\n\n");

    /* Create synthetic Q4_K block */
    block_q4_K test_block;
    memset(&test_block, 0, sizeof(test_block));

    /* Set scale d=1.0 in FP16 (0x3C00) and dmin=0 */
    test_block.d = 0x3C00;     /* FP16 1.0 */
    test_block.dmin = 0x0000;  /* FP16 0.0 */

    /* Set all scales to 1 (6-bit packed) */
    /* Simple: just set first scale byte */
    test_block.scales[0] = 1;  /* scale[0] = 1 */

    /* Set some q4 values: alternating 0 and 15 */
    int i;
    for (i = 0; i < 128; i++) {
        test_block.qs[i] = 0x0F;  /* low nibble=15, high nibble=0 */
    }

    /* Test input vector: all ones */
    float x[256];
    for (i = 0; i < 256; i++) x[i] = 1.0f;

    float dot = q4k_dot_block(&test_block, x);
    printf("Synthetic test: dot = %.4f (expected: 240.0 for 16 values of 15*1)\n", dot);

    /* Benchmark: matmul with synthetic data */
    /* Create a 1024 x 256 matrix (1024 rows, 1 block per row) */
    int out_dim = 1024;
    int in_dim = 256;
    int n_blocks = out_dim;  /* 1 block per row */

    block_q4_K* W = (block_q4_K*)_aligned_malloc(n_blocks * sizeof(block_q4_K), 64);
    float* y = (float*)_aligned_malloc(out_dim * sizeof(float), 32);

    /* Fill with test data */
    for (i = 0; i < n_blocks; i++) {
        W[i] = test_block;
    }

    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);

    /* Benchmark small matmul */
    printf("\nBenchmark: 1024 x 256 matmul...\n");
    QueryPerformanceCounter(&start);
    for (i = 0; i < 1000; i++) {
        q4k_matvec(y, W, x, out_dim, in_dim);
    }
    QueryPerformanceCounter(&end);
    double ms = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart * 1000.0;
    printf("  1000 iters: %.2f ms (%.3f ms/iter)\n", ms, ms / 1000);

    /* Benchmark realistic size: 512 x 2048 (35B expert gate_proj) */
    printf("\nBenchmark: 512 x 2048 (35B gate_proj)...\n");
    int out2 = 512, in2 = 2048;
    int n_blocks2 = out2 * (in2 / Q4K_QK);
    block_q4_K* W2 = (block_q4_K*)_aligned_malloc(n_blocks2 * sizeof(block_q4_K), 64);
    float* x2 = (float*)_aligned_malloc(in2 * sizeof(float), 32);
    float* y2 = (float*)_aligned_malloc(out2 * sizeof(float), 32);

    for (i = 0; i < n_blocks2; i++) W2[i] = test_block;
    for (i = 0; i < in2; i++) x2[i] = 0.01f;

    QueryPerformanceCounter(&start);
    q4k_matvec(y2, W2, x2, out2, in2);
    QueryPerformanceCounter(&end);
    ms = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart * 1000.0;
    printf("  gate_proj (512x2048): %.3f ms\n", ms);
    printf("  per expert (gate+up+down): %.3f ms\n", ms * 3);
    printf("  per layer K=8: %.3f ms\n", ms * 3 * 8);
    printf("  40 layers: %.2f ms = %.2f tok/s\n", ms * 3 * 8 * 40, 1000.0 / (ms * 3 * 8 * 40));

    /* Benchmark 397B size: 1024 x 4096 */
    printf("\nBenchmark: 1024 x 4096 (397B gate_proj)...\n");
    int out3 = 1024, in3 = 4096;
    int n_blocks3 = out3 * (in3 / Q4K_QK);
    block_q4_K* W3 = (block_q4_K*)_aligned_malloc(n_blocks3 * sizeof(block_q4_K), 64);
    float* x3 = (float*)_aligned_malloc(in3 * sizeof(float), 32);
    float* y3 = (float*)_aligned_malloc(out3 * sizeof(float), 32);

    for (i = 0; i < n_blocks3; i++) W3[i] = test_block;
    for (i = 0; i < in3; i++) x3[i] = 0.01f;

    QueryPerformanceCounter(&start);
    q4k_matvec(y3, W3, x3, out3, in3);
    QueryPerformanceCounter(&end);
    ms = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart * 1000.0;
    printf("  gate_proj (1024x4096): %.3f ms\n", ms);
    printf("  per expert K=3: %.3f ms\n", ms * 3 * 3);
    printf("  60 layers: %.2f ms = %.2f tok/s\n", ms * 3 * 3 * 60, 1000.0 / (ms * 3 * 3 * 60));

    _aligned_free(W); _aligned_free(y);
    _aligned_free(W2); _aligned_free(x2); _aligned_free(y2);
    _aligned_free(W3); _aligned_free(x3); _aligned_free(y3);
    printf("\n=== DONE ===\n");
    return 0;
}
