/*
 * Full Expert FFN Test on Qwen3.5-35B-A3B Q4_K_M
 *
 * Parses GGUF, finds expert tensors, reads via explicit I/O,
 * runs gate→SwiGLU→down pipeline, measures tok/s.
 *
 * 35B model config:
 *   hidden_dim = 2048
 *   expert_intermediate = 512
 *   num_experts = 256
 *   K = 8 (active per layer)
 *   num_layers = 40
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <windows.h>

#include "iq2xxs_grid_def.h"
#include "iq2_dequant.h"
#include "q4k_dequant.h"
#include "q5k_dequant.h"
#include "gguf_parser.h"

#define GGUF_PATH "D:/models/qwen35-35b-q4/Qwen3.5-35B-A3B-Q4_K_M.gguf"

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

int main(void) {
    printf("=== Qwen3.5-35B-A3B Q4_K_M — Full Expert FFN Test ===\n\n");

    /* Parse GGUF */
    GGUFModel model;
    if (parse_gguf(GGUF_PATH, &model) != 0) {
        printf("ERROR: Cannot parse GGUF at %s\n", GGUF_PATH);
        printf("(Model may still be downloading)\n");
        return 1;
    }

    printf("Model: v%u, %llu tensors\n", model.version, model.n_tensors);
    printf("Config: hidden=%d, intermediate=%d, experts=%d, layers=%d, K=%d\n",
           model.hidden_dim, model.expert_intermediate,
           model.num_experts, model.num_layers, model.expert_used_count);
    printf("Data starts at offset: %llu\n\n", model.data_start);

    /* Find expert tensors for layer 0 */
    TensorInfo* gate = find_tensor(&model, "blk.0.ffn_gate_exps.weight");
    TensorInfo* up = find_tensor(&model, "blk.0.ffn_up_exps.weight");
    TensorInfo* down = find_tensor(&model, "blk.0.ffn_down_exps.weight");

    if (!gate) { printf("ERROR: gate_exps not found\n"); return 1; }
    if (!up)   { printf("ERROR: up_exps not found\n"); return 1; }
    if (!down) { printf("ERROR: down_exps not found\n"); return 1; }

    printf("Layer 0 expert tensors:\n");
    printf("  gate: dims=[%llu,%llu,%llu], type=%d, offset=%llu, size=%llu\n",
           gate->dims[0], gate->dims[1], gate->dims[2], gate->type,
           gate->offset, gate->data_size);
    printf("  up:   dims=[%llu,%llu,%llu], type=%d, offset=%llu, size=%llu\n",
           up->dims[0], up->dims[1], up->dims[2], up->type,
           up->offset, up->data_size);
    printf("  down: dims=[%llu,%llu,%llu], type=%d, offset=%llu, size=%llu\n",
           down->dims[0], down->dims[1], down->dims[2], down->type,
           down->offset, down->data_size);

    /* Calculate per-expert data size */
    int n_experts = (int)gate->dims[2]; /* third dim is num_experts */
    uint64_t gate_per_expert = gate->data_size / n_experts;
    uint64_t up_per_expert = up->data_size / n_experts;
    uint64_t down_per_expert = down->data_size / n_experts;

    printf("\nPer expert:\n");
    printf("  gate: %llu bytes\n", gate_per_expert);
    printf("  up:   %llu bytes\n", up_per_expert);
    printf("  down: %llu bytes\n", down_per_expert);
    printf("  total: %llu bytes\n", gate_per_expert + up_per_expert + down_per_expert);

    /* Verify Q4_K block alignment */
    printf("\nQ4_K block check:\n");
    int gate_blocks = (int)(gate_per_expert / Q4K_BLOCK_SIZE);
    int gate_weights = gate_blocks * Q4K_QK;
    printf("  gate: %d blocks * %d = %d weights (expected %llu)\n",
           gate_blocks, Q4K_QK, gate_weights, gate->dims[0] * gate->dims[1]);
    printf("  remainder: %llu bytes\n", gate_per_expert % Q4K_BLOCK_SIZE);

    /* Open GGUF for direct I/O reading */
    HANDLE hFile = CreateFileA(GGUF_PATH, GENERIC_READ, FILE_SHARE_READ,
        NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        printf("ERROR: Cannot open GGUF for direct I/O\n");
        return 1;
    }

    /* Read expert 0 gate weights */
    uint64_t expert_offset = model.data_start + gate->offset; /* expert 0 */
    int align = 65536;
    int buf_size = ((int)gate_per_expert + align * 2);
    buf_size = ((buf_size + align - 1) / align) * align;
    void* buf = _aligned_malloc(buf_size, align);

    uint64_t aligned_off = (expert_offset / align) * align;
    int sub_off = (int)(expert_offset - aligned_off);

    LARGE_INTEGER li;
    li.QuadPart = aligned_off;
    SetFilePointerEx(hFile, li, NULL, FILE_BEGIN);
    DWORD br;
    ReadFile(hFile, buf, buf_size, &br, NULL);

    void* gate_data = (char*)buf + sub_off;
    printf("\nRead gate expert 0: %u bytes from offset %llu\n", br, expert_offset);

    /* Verify first block */
    block_q4_K* first_block = (block_q4_K*)gate_data;
    printf("  d=0x%04x (%.6f), dmin=0x%04x (%.6f)\n",
           first_block->d, fp16_to_fp32(first_block->d),
           first_block->dmin, fp16_to_fp32(first_block->dmin));

    /* Run gate matmul */
    int hidden = model.hidden_dim;
    int intermediate = model.expert_intermediate;
    float* x = (float*)_aligned_malloc(hidden * sizeof(float), 32);
    float* gate_out = (float*)_aligned_malloc(intermediate * sizeof(float), 32);
    int i;
    for (i = 0; i < hidden; i++) x[i] = 0.01f;

    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);

    printf("\nGate matmul (%d x %d, Q4_K)...\n", intermediate, hidden);
    QueryPerformanceCounter(&start);
    q4k_matvec(gate_out, gate_data, x, intermediate, hidden);
    QueryPerformanceCounter(&end);
    double ms = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart * 1000.0;
    printf("  Time: %.3f ms\n", ms);
    printf("  gate[0..3] = %.4f %.4f %.4f %.4f\n",
           gate_out[0], gate_out[1], gate_out[2], gate_out[3]);

    /* Check for NaN */
    int has_nan = 0;
    for (i = 0; i < intermediate; i++) if (gate_out[i] != gate_out[i]) has_nan = 1;
    printf("  NaN: %s\n", has_nan ? "YES" : "no");

    /* Project full model */
    printf("\nProjections (35B, K=8, 40 layers):\n");
    printf("  Per expert (3 matmuls): %.3f ms\n", ms * 3);
    printf("  Per layer K=8: %.3f ms\n", ms * 3 * 8);
    printf("  40 layers: %.2f ms = %.2f tok/s (compute only)\n",
           ms * 3 * 8 * 40, 1000.0 / (ms * 3 * 8 * 40));

    /* === FULL EXPERT FFN: gate + up → SwiGLU → down === */
    printf("\n=== Full Expert FFN (gate Q4_K + up Q4_K + down Q5_K) ===\n");

    /* Read up expert 0 */
    uint64_t up_offset = model.data_start + up->offset;
    uint64_t up_aligned = (up_offset / align) * align;
    int up_sub = (int)(up_offset - up_aligned);
    void* up_buf = _aligned_malloc(buf_size, align);
    li.QuadPart = up_aligned;
    SetFilePointerEx(hFile, li, NULL, FILE_BEGIN);
    ReadFile(hFile, up_buf, buf_size, &br, NULL);
    void* up_data = (char*)up_buf + up_sub;

    /* Read down expert 0 */
    uint64_t down_offset = model.data_start + down->offset;
    uint64_t down_aligned = (down_offset / align) * align;
    int down_sub = (int)(down_offset - down_aligned);
    int down_buf_size = ((int)down_per_expert + align * 2);
    down_buf_size = ((down_buf_size + align - 1) / align) * align;
    void* down_buf = _aligned_malloc(down_buf_size, align);
    li.QuadPart = down_aligned;
    SetFilePointerEx(hFile, li, NULL, FILE_BEGIN);
    ReadFile(hFile, down_buf, down_buf_size, &br, NULL);
    void* down_data = (char*)down_buf + down_sub;

    float* up_out = (float*)_aligned_malloc(intermediate * sizeof(float), 32);
    float* act = (float*)_aligned_malloc(intermediate * sizeof(float), 32);
    float* ffn_out = (float*)_aligned_malloc(hidden * sizeof(float), 32);

    /* Warm up */
    q4k_matvec(gate_out, gate_data, x, intermediate, hidden);
    q4k_matvec(up_out, up_data, x, intermediate, hidden);
    for (i = 0; i < intermediate; i++) act[i] = silu(gate_out[i]) * up_out[i];
    q5k_matvec(ffn_out, down_data, act, hidden, intermediate);

    /* Benchmark full FFN */
    printf("Single expert FFN:\n");
    QueryPerformanceCounter(&start);
    q4k_matvec(gate_out, gate_data, x, intermediate, hidden);
    q4k_matvec(up_out, up_data, x, intermediate, hidden);
    for (i = 0; i < intermediate; i++) act[i] = silu(gate_out[i]) * up_out[i];
    q5k_matvec(ffn_out, down_data, act, hidden, intermediate);
    QueryPerformanceCounter(&end);
    ms = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart * 1000.0;
    printf("  Time: %.3f ms\n", ms);
    printf("  ffn_out[0..3] = %.6f %.6f %.6f %.6f\n",
           ffn_out[0], ffn_out[1], ffn_out[2], ffn_out[3]);
    has_nan = 0;
    for (i = 0; i < hidden; i++) if (ffn_out[i] != ffn_out[i]) has_nan = 1;
    printf("  NaN: %s\n", has_nan ? "YES" : "no");

    /* 10-iter average */
    printf("\n10-iteration average (full FFN):\n");
    QueryPerformanceCounter(&start);
    for (i = 0; i < 10; i++) {
        q4k_matvec(gate_out, gate_data, x, intermediate, hidden);
        q4k_matvec(up_out, up_data, x, intermediate, hidden);
        int j;
        for (j = 0; j < intermediate; j++) act[j] = silu(gate_out[j]) * up_out[j];
        q5k_matvec(ffn_out, down_data, act, hidden, intermediate);
    }
    QueryPerformanceCounter(&end);
    ms = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart * 1000.0 / 10;
    printf("  Avg per expert: %.3f ms\n", ms);
    printf("  Per layer K=8: %.3f ms\n", ms * 8);
    printf("  40 layers: %.2f ms = %.2f tok/s (compute only)\n",
           ms * 8 * 40, 1000.0 / (ms * 8 * 40));
    printf("  With I/O streaming (7%%): %.2f ms = %.2f tok/s\n",
           ms * 8 * 40 * 1.07, 1000.0 / (ms * 8 * 40 * 1.07));

    _aligned_free(x);
    _aligned_free(gate_out);
    _aligned_free(up_out);
    _aligned_free(act);
    _aligned_free(ffn_out);
    _aligned_free(buf);
    _aligned_free(up_buf);
    _aligned_free(down_buf);
    CloseHandle(hFile);

    printf("\n=== DONE ===\n");
    return 0;
}
