/*
 * Test: Complete Expert FFN from slab data
 * Measures end-to-end time for: read expert from slab → dequant → FFN → output
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <windows.h>

#include "iq2xxs_grid_def.h"
#include "iq2_dequant.h"
#include "iq2_dequant_fast.h"
#include "expert_ffn.h"

#define SLAB_PATH "D:/flash-moe-engine/experts.slab"
#define SLOT_SIZE 3735552
#define ALIGNMENT 65536
/* Actual IQ2_XXS data sizes (4096x1024 = 16384 blocks * 66 bytes) */
#define GATE_DATA_SIZE 1081344
#define GATE_SLAB_SIZE 1103392  /* includes 22048 padding */
#define UP_DATA_SIZE   1081344  /* same shape as gate */
#define UP_SLAB_SIZE   1254914  /* from expert_index.json - different padding? */

int main(void) {
    printf("=== Expert FFN End-to-End Test ===\n\n");

    /* Open slab */
    HANDLE hFile = CreateFileA(SLAB_PATH, GENERIC_READ, FILE_SHARE_READ,
        NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        printf("ERROR: cannot open slab\n"); return 1;
    }

    /* Read dense expert: layer 2, expert 511 (94.4% nonzero) */
    int expert_slot = 2 * 512 + 511;  /* layer 2, expert 511 */
    int buf_size = ((SLOT_SIZE + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    void* buf = _aligned_malloc(buf_size, ALIGNMENT);
    LARGE_INTEGER seek;
    seek.QuadPart = ((long long)expert_slot * SLOT_SIZE / ALIGNMENT) * ALIGNMENT;
    SetFilePointerEx(hFile, seek, NULL, FILE_BEGIN);
    DWORD br;
    ReadFile(hFile, buf, buf_size, &br, NULL);
    int sub_off = (int)((long long)expert_slot * SLOT_SIZE - seek.QuadPart);
    void* expert_data = (char*)buf + sub_off;
    printf("Using dense expert: layer 2, expert 511 (94.4%% nonzero)\n\n");

    /* Allocate buffers */
    float* x = (float*)_aligned_malloc(HIDDEN_DIM * sizeof(float), 32);
    float* out = (float*)_aligned_malloc(HIDDEN_DIM * sizeof(float), 32);
    float* gate_buf = (float*)malloc(EXPERT_INTERMEDIATE * sizeof(float));
    float* up_buf = (float*)malloc(EXPERT_INTERMEDIATE * sizeof(float));
    float* act_buf = (float*)malloc(EXPERT_INTERMEDIATE * sizeof(float));
    int i;

    /* Initialize input */
    for (i = 0; i < HIDDEN_DIM; i++) x[i] = 0.01f;

    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);

    /* Test gate matmul only — verify non-NaN output */
    printf("Gate-only matmul (1024 x 4096, IQ2_XXS):\n");
    iq2xxs_matvec_fast(gate_buf, expert_data, x, 1024, 4096);
    printf("  gate[0..3] = %.4f %.4f %.4f %.4f\n",
           gate_buf[0], gate_buf[1], gate_buf[2], gate_buf[3]);
    int has_nan = 0;
    for (i = 0; i < 1024; i++) if (gate_buf[i] != gate_buf[i]) has_nan = 1;
    printf("  Has NaN: %s\n", has_nan ? "YES" : "no");
    printf("  gate[100..103] = %.4f %.4f %.4f %.4f\n",
           gate_buf[100], gate_buf[101], gate_buf[102], gate_buf[103]);

    /* Benchmark: single expert FFN */
    printf("\nSingle expert FFN (gate+up→SwiGLU→down):\n");
    expert_ffn(out, x, expert_data, gate_buf, up_buf, act_buf, GATE_SLAB_SIZE, UP_SLAB_SIZE);
    QueryPerformanceCounter(&start);
    expert_ffn(out, x, expert_data, gate_buf, up_buf, act_buf, GATE_SLAB_SIZE, UP_SLAB_SIZE);
    QueryPerformanceCounter(&end);
    double ms1 = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart * 1000.0;
    printf("  Time: %.2f ms\n", ms1);
    printf("  out[0..3] = %.4f %.4f %.4f %.4f\n", out[0], out[1], out[2], out[3]);

    /* Benchmark: K=3 experts (simulate MoE layer) */
    printf("\nK=3 experts (one MoE layer):\n");
    float expert_outs[3][HIDDEN_DIM];
    float routing_w[3] = {0.5f, 0.3f, 0.2f};

    QueryPerformanceCounter(&start);
    for (i = 0; i < 3; i++) {
        expert_ffn(expert_outs[i], x, expert_data, gate_buf, up_buf, act_buf,
                   GATE_SLAB_SIZE, UP_SLAB_SIZE);
    }
    moe_combine(out, x, (const float(*)[HIDDEN_DIM])expert_outs, routing_w, 3);
    QueryPerformanceCounter(&end);
    double ms3 = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart * 1000.0;
    printf("  Time: %.2f ms (%.2f ms/expert)\n", ms3, ms3 / 3);

    /* Project to full model */
    printf("\nProjections:\n");
    printf("  Per layer (K=3): %.2f ms\n", ms3);
    printf("  60 layers: %.2f ms = %.2f tok/s (compute only)\n",
           ms3 * 60, 1000.0 / (ms3 * 60));

    /* With I/O streaming */
    double io_overhead_pct = 0.068; /* 6.8% from simulator */
    double total_ms = ms3 * 60 * (1 + io_overhead_pct);
    printf("  With I/O streaming (6.8%% overhead): %.2f ms = %.2f tok/s\n",
           total_ms, 1000.0 / total_ms);

    /* Run 10 iterations for stable measurement */
    printf("\n10-iteration average:\n");
    QueryPerformanceCounter(&start);
    for (int iter = 0; iter < 10; iter++) {
        for (i = 0; i < 3; i++) {
            expert_ffn(expert_outs[i], x, expert_data, gate_buf, up_buf, act_buf,
                       GATE_SLAB_SIZE, UP_SLAB_SIZE);
        }
        moe_combine(out, x, (const float(*)[HIDDEN_DIM])expert_outs, routing_w, 3);
    }
    QueryPerformanceCounter(&end);
    double ms_avg = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart * 1000.0 / 10;
    printf("  Avg per layer: %.2f ms (%.2f ms/expert)\n", ms_avg, ms_avg / 3);
    printf("  60 layers: %.2f ms = %.2f tok/s\n", ms_avg * 60, 1000.0 / (ms_avg * 60));
    total_ms = ms_avg * 60 * (1 + io_overhead_pct);
    printf("  With I/O streaming: %.2f ms = %.2f tok/s\n", total_ms, 1000.0 / total_ms);

    /* Cleanup */
    _aligned_free(x);
    _aligned_free(out);
    free(gate_buf); free(up_buf); free(act_buf);
    _aligned_free(buf);
    CloseHandle(hFile);

    printf("\n=== DONE ===\n");
    return 0;
}
