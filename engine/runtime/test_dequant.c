/*
 * Test IQ2_XXS dequantization against slab data
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <windows.h>

/* ---- Grid LUT (from llama.cpp) ---- */
#include "iq2xxs_grid_def.h"

/* ---- Dequant implementation ---- */
#include "iq2_dequant.h"
#include "iq2_dequant_avx.h"
#include "iq2_dequant_fast.h"

#define SLAB_PATH "D:/flash-moe-engine/experts.slab"
#define SLOT_SIZE 3735552
#define ALIGNMENT 65536

int main(void) {
    printf("=== IQ2_XXS Dequant Test ===\n\n");

    HANDLE hFile = CreateFileA(SLAB_PATH, GENERIC_READ, FILE_SHARE_READ,
        NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        printf("ERROR: cannot open slab\n");
        return 1;
    }

    int buf_size = ((SLOT_SIZE + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    void* buf = _aligned_malloc(buf_size, ALIGNMENT);

    /* Seek to layer 0, expert 21 (first non-zero expert) */
    LARGE_INTEGER seek_pos;
    seek_pos.QuadPart = ((long long)21 * SLOT_SIZE / ALIGNMENT) * ALIGNMENT;
    SetFilePointerEx(hFile, seek_pos, NULL, FILE_BEGIN);

    DWORD bytes_read;
    ReadFile(hFile, buf, buf_size, &bytes_read, NULL);
    CloseHandle(hFile);

    /* Adjust for sub-alignment offset */
    int sub_off = (int)((long long)21 * SLOT_SIZE - seek_pos.QuadPart);
    void* expert_data = (char*)buf + sub_off;

    printf("Read %u bytes from slab\n", bytes_read);

    const block_iq2_xxs* block = (const block_iq2_xxs*)expert_data;
    printf("First block d (FP16): 0x%04x = %.6f\n", block->d, fp16_to_fp32(block->d));

    float x[256];
    int i;
    for (i = 0; i < 256; i++) x[i] = 1.0f;

    float dot = iq2xxs_dot_block(block, x);
    printf("dot(block, ones) = %.6f\n\n", dot);

    /* Benchmark: expert FFN gate_proj (1024 x 4096) */
    float* big_x = (float*)malloc(4096 * sizeof(float));
    float* big_y = (float*)malloc(1024 * sizeof(float));
    for (i = 0; i < 4096; i++) big_x[i] = 0.01f;

    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);

    printf("Benchmark: gate_proj (1024 x 4096)...\n");
    QueryPerformanceCounter(&start);
    iq2xxs_matvec(big_y, expert_data, big_x, 1024, 4096);
    QueryPerformanceCounter(&end);

    double ms = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart * 1000.0;
    printf("  gate_proj: %.2f ms\n", ms);
    printf("  per expert (gate+up+down): %.2f ms\n", ms * 3);
    printf("  per layer K=3: %.2f ms\n", ms * 9);
    printf("  60 layers: %.2f ms = %.2f tok/s (compute only)\n",
           ms * 9 * 60, 1000.0 / (ms * 9 * 60));

    printf("  y[0..4] = %.4f %.4f %.4f %.4f %.4f\n",
           big_y[0], big_y[1], big_y[2], big_y[3], big_y[4]);

    /* AVX2 benchmark */
    printf("\nBenchmark: gate_proj AVX2 (1024 x 4096)...\n");
    float* big_y2 = (float*)malloc(1024 * sizeof(float));
    QueryPerformanceCounter(&start);
    iq2xxs_matvec_avx2(big_y2, expert_data, big_x, 1024, 4096);
    QueryPerformanceCounter(&end);
    ms = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart * 1000.0;
    printf("  gate_proj AVX2: %.2f ms (%.1fx vs scalar)\n", ms, 4.19 / ms);
    printf("  per expert: %.2f ms\n", ms * 3);
    printf("  per layer K=3: %.2f ms\n", ms * 9);
    printf("  60 layers: %.2f ms = %.2f tok/s\n", ms * 9 * 60, 1000.0 / (ms * 9 * 60));
    printf("  y2[0..4] = %.4f %.4f %.4f %.4f %.4f\n",
           big_y2[0], big_y2[1], big_y2[2], big_y2[3], big_y2[4]);

    /* Fast (unrolled + OpenMP) benchmark */
    printf("\nBenchmark: gate_proj FAST (1024 x 4096)...\n");
    float* big_y3 = (float*)malloc(1024 * sizeof(float));
    QueryPerformanceCounter(&start);
    iq2xxs_matvec_fast(big_y3, expert_data, big_x, 1024, 4096);
    QueryPerformanceCounter(&end);
    ms = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart * 1000.0;
    printf("  gate_proj FAST: %.2f ms (%.1fx vs scalar)\n", ms, 4.19 / ms);
    printf("  per expert: %.2f ms\n", ms * 3);
    printf("  60 layers K=3: %.2f ms = %.2f tok/s\n", ms * 9 * 60, 1000.0 / (ms * 9 * 60));
    free(big_y3);

    /* Verify AVX2 matches scalar */
    float max_diff = 0;
    for (i = 0; i < 1024; i++) {
        float diff = (float)fabs(big_y[i] - big_y2[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  Max scalar vs AVX2 diff: %.6f %s\n", max_diff,
           max_diff < 1.0f ? "OK" : "MISMATCH!");

    free(big_y2);
    free(big_x);
    free(big_y);
    _aligned_free(buf);
    printf("\n=== DONE ===\n");
    return 0;
}
