/*
 * GPU Offload v2 — Phase 3 Pipeline Architecture
 *
 * 3 CUDA streams: compute, H2D, D2H
 * Double-buffered pinned memory slots
 * Custom Q8_0 matvec kernel (weights stay in Q8_0 on GPU)
 * Async API: launch → do CPU work → wait
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpu_offload.h"

#define MAX_LAYERS 64
#define MATVEC_THREADS 256
#define NUM_PIPELINE_SLOTS 2

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

/* Per-layer GPU weight pointers (Q8_0 on device) */
typedef struct {
    /* DeltaNet weights */
    void* d_qkv;
    void* d_gate;
    void* d_ssm_out;
    int qkv_rows, qkv_cols;
    int gate_rows, gate_cols;
    int ssm_rows, ssm_cols;
    int loaded;
    /* Standard GQA weights (Q8_0) */
    void* d_wq;     /* [hidden, nqh*hd*2] — doubled for gate */
    void* d_wk;     /* [hidden, nkvh*hd] */
    void* d_wv;     /* [hidden, nkvh*hd] */
    void* d_wo;     /* [nqh*hd, hidden] */
    int wq_rows, wq_cols;
    int wk_rows, wk_cols;
    int wv_rows, wv_cols;
    int wo_rows, wo_cols;
    int gqa_loaded;
} LayerGPU;

/* Double-buffered pipeline slot */
typedef struct {
    /* Pinned host memory */
    float* h_normed;
    float* h_qkv_out;
    float* h_gate_out;
    float* h_gated_in;
    float* h_ssm_out;
    /* Device memory */
    float* d_normed;
    float* d_qkv_out;
    float* d_gate_out;
    float* d_gated_in;
    float* d_ssm_out;
    /* CUDA events */
    cudaEvent_t evt_h2d_normed_done;
    cudaEvent_t evt_qkv_gate_done;
    cudaEvent_t evt_d2h_qkv_gate_done;
    cudaEvent_t evt_h2d_gated_done;
    cudaEvent_t evt_ssm_out_done;
    cudaEvent_t evt_d2h_ssm_out_done;
} PipelineSlot;

static cublasHandle_t g_cublas = NULL;
static LayerGPU g_layers[MAX_LAYERS];
static PipelineSlot g_slots[NUM_PIPELINE_SLOTS];
static cudaStream_t g_stream_compute = NULL;
static cudaStream_t g_stream_h2d = NULL;
static cudaStream_t g_stream_d2h = NULL;
static cudaStream_t g_stream_expert = NULL;  /* Dedicated stream for expert FFN */
static int g_initialized = 0;
static float g_vram_used = 0;
static int g_hidden_dim = 4096;
static int g_qkv_dim = 8192;
static int g_gate_dim = 4096;

/* GGML Q4_K/Q5_K scale decode helper (matches get_scale_min_k4) */
__device__ __forceinline__ void gpu_get_scale_min(int sub, const unsigned char* sc, int* scale, int* mn) {
    if (sub < 4) {
        *scale = sc[sub] & 63;
        *mn = sc[sub + 4] & 63;
    } else {
        *scale = (sc[sub + 4] & 0xF) | ((sc[sub - 4] >> 6) << 4);
        *mn = (sc[sub + 4] >> 4) | ((sc[sub] >> 6) << 4);
    }
}

/* Q8_0 matvec kernel — shared memory input caching + vectorized loads */
__global__ void q8_matvec_kernel(float* output, const unsigned char* weights,
                                  const float* input, int in_dim, int out_dim) {
    int row = blockIdx.x;
    if (row >= out_dim) return;
    int tid = threadIdx.x;
    int blocks_per_row = in_dim / 32;
    const unsigned char* row_data = weights + (long long)row * blocks_per_row * 34;

    /* Cache input vector in shared memory (all threads collaborate to load) */
    extern __shared__ float s_input[];
    for (int i = tid; i < in_dim; i += MATVEC_THREADS)
        s_input[i] = input[i];
    __syncthreads();

    float partial = 0.0f;
    for (int b = tid; b < blocks_per_row; b += MATVEC_THREADS) {
        const unsigned char* block = row_data + b * 34;
        half d_h;
        memcpy(&d_h, block, 2);
        float d = __half2float(d_h);
        const signed char* qs = (const signed char*)(block + 2);
        const float* xp = s_input + b * 32; /* read from shared memory */

        float block_sum = 0.0f;
        for (int j = 0; j < 32; j += 4) {
            block_sum += (float)qs[j] * xp[j] + (float)qs[j+1] * xp[j+1] +
                         (float)qs[j+2] * xp[j+2] + (float)qs[j+3] * xp[j+3];
        }
        partial += d * block_sum;
    }

    /* Warp reduction */
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);

    /* Block reduction via shared memory */
    __shared__ float shared[8];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) shared[warp_id] = partial;
    __syncthreads();
    if (tid < 8) {
        float val = shared[tid];
        for (int offset = 4; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFF, val, offset);
        if (tid == 0) output[row] = val;
    }
}

/*
 * Q4_K matvec CUDA kernel — nibble unpacking + dot product
 * Q4_K block: 144 bytes per 256 weights
 *   d(2) + dmin(2) + scales(12) + qs(128)
 */
__global__ void q4k_matvec_kernel(float* output, const unsigned char* weights,
                                   const float* input, int in_dim, int out_dim) {
    int row = blockIdx.x;
    if (row >= out_dim) return;
    int tid = threadIdx.x;
    int blocks_per_row = in_dim / 256;
    const unsigned char* row_data = weights + (long long)row * blocks_per_row * 144;

    /* Cache input in shared memory */
    extern __shared__ float s_input[];
    for (int i = tid; i < in_dim; i += MATVEC_THREADS)
        s_input[i] = input[i];
    __syncthreads();

    float partial = 0.0f;
    for (int b = tid; b < blocks_per_row; b += MATVEC_THREADS) {
        const unsigned char* blk = row_data + b * 144;
        half d_h, dmin_h;
        memcpy(&d_h, blk, 2);
        memcpy(&dmin_h, blk + 2, 2);
        float d = __half2float(d_h);
        float dmin = __half2float(dmin_h);

        /* Decode 6-bit scales/mins from packed 12 bytes */
        const unsigned char* sc = blk + 4;
        const unsigned char* qs = blk + 16;

        float block_sum = 0.0f;
        for (int sub = 0; sub < 8; sub++) {
            int scale_val, min_val;
            gpu_get_scale_min(sub, sc, &scale_val, &min_val);
            float sc_f = d * (float)scale_val;
            float mn_f = dmin * (float)min_val;

            for (int j = 0; j < 16; j++) {
                unsigned char packed = qs[sub * 16 + j];
                /* GGML Q4_K: low nibble = weight[j], high nibble = weight[j+16] (BLOCKED, not interleaved) */
                int idx0 = sub * 32 + j;       /* first 16 weights */
                int idx1 = sub * 32 + j + 16;  /* second 16 weights */
                float w0 = sc_f * (float)(packed & 0x0F) - mn_f;
                float w1 = sc_f * (float)(packed >> 4) - mn_f;
                block_sum += w0 * s_input[b * 256 + idx0] + w1 * s_input[b * 256 + idx1];
            }
        }
        partial += block_sum;
    }

    /* Warp + block reduction (same as Q8_0) */
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
    __shared__ float shared[8];
    int warp_id = tid / 32, lane_id = tid % 32;
    if (lane_id == 0) shared[warp_id] = partial;
    __syncthreads();
    if (tid < 8) {
        float val = shared[tid];
        for (int offset = 4; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFF, val, offset);
        if (tid == 0) output[row] = val;
    }
}

/* Q5_K matvec CUDA kernel — Q4_K + high bit per weight */
__global__ void q5k_matvec_kernel(float* output, const unsigned char* weights,
                                   const float* input, int in_dim, int out_dim) {
    int row = blockIdx.x;
    if (row >= out_dim) return;
    int tid = threadIdx.x;
    int blocks_per_row = in_dim / 256;
    const unsigned char* row_data = weights + (long long)row * blocks_per_row * 176;

    extern __shared__ float s_input[];
    for (int i = tid; i < in_dim; i += MATVEC_THREADS)
        s_input[i] = input[i];
    __syncthreads();

    float partial = 0.0f;
    for (int b = tid; b < blocks_per_row; b += MATVEC_THREADS) {
        const unsigned char* blk = row_data + b * 176;
        half d_h, dmin_h;
        memcpy(&d_h, blk, 2);
        memcpy(&dmin_h, blk + 2, 2);
        float d = __half2float(d_h);
        float dmin = __half2float(dmin_h);

        const unsigned char* sc = blk + 4;
        const unsigned char* qh = blk + 16;  /* 32 bytes of high bits */
        const unsigned char* qs = blk + 48;  /* 128 bytes of low nibbles */

        float block_sum = 0.0f;
        for (int sub = 0; sub < 8; sub++) {
            int scale_val, min_val;
            gpu_get_scale_min(sub, sc, &scale_val, &min_val);
            float sc_f = d * (float)scale_val;
            float mn_f = dmin * (float)min_val;

            for (int j = 0; j < 16; j++) {
                unsigned char packed = qs[sub * 16 + j];
                /* GGML Q5_K: low nibble = weight[j], high nibble = weight[j+16] (BLOCKED) */
                int idx0 = sub * 32 + j;       /* first 16 weights */
                int idx1 = sub * 32 + j + 16;  /* second 16 weights */
                int q0 = (packed & 0x0F) | (((qh[idx0 / 8] >> (idx0 % 8)) & 1) << 4);
                int q1 = (packed >> 4) | (((qh[idx1 / 8] >> (idx1 % 8)) & 1) << 4);
                float w0 = sc_f * (float)q0 - mn_f;
                float w1 = sc_f * (float)q1 - mn_f;
                block_sum += w0 * s_input[b * 256 + idx0] + w1 * s_input[b * 256 + idx1];
            }
        }
        partial += block_sum;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
    __shared__ float shared_q5[8];
    int warp_id = tid / 32, lane_id = tid % 32;
    if (lane_id == 0) shared_q5[warp_id] = partial;
    __syncthreads();
    if (tid < 8) {
        float val = shared_q5[tid];
        for (int offset = 4; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFF, val, offset);
        if (tid == 0) output[row] = val;
    }
}

/*
 * Expert-optimized Q4_K kernel — 32 threads, sub-block work distribution
 * For small in_dim (e.g. 1024-4096), the standard kernel wastes 94% of threads.
 * This kernel distributes work at sub-block granularity (blocks_per_row * 8 units)
 * and uses warp-only reduction (no block reduction needed for 32 threads).
 */
#define EXPERT_THREADS 32

__global__ void q4k_expert_kernel(float* output, const unsigned char* weights,
                                   const float* input, int in_dim, int out_dim) {
    int row = blockIdx.x;
    if (row >= out_dim) return;
    int tid = threadIdx.x;
    int blocks_per_row = in_dim / 256;
    const unsigned char* row_data = weights + (long long)row * blocks_per_row * 144;

    /* Cache input in shared memory — 32 threads load cooperatively */
    extern __shared__ float s_input[];
    for (int i = tid; i < in_dim; i += EXPERT_THREADS)
        s_input[i] = input[i];
    __syncthreads();

    /* Distribute at sub-block level: blocks_per_row * 8 work units */
    float partial = 0.0f;
    int total_subs = blocks_per_row * 8;

    for (int si = tid; si < total_subs; si += EXPERT_THREADS) {
        int b = si / 8;
        int sub = si % 8;

        const unsigned char* blk = row_data + b * 144;
        half d_h, dmin_h;
        memcpy(&d_h, blk, 2);
        memcpy(&dmin_h, blk + 2, 2);
        float d = __half2float(d_h);
        float dmin = __half2float(dmin_h);

        const unsigned char* sc = blk + 4;
        const unsigned char* qs = blk + 16;

        int scale_val, min_val;
        gpu_get_scale_min(sub, sc, &scale_val, &min_val);
        float sc_f = d * (float)scale_val;
        float mn_f = dmin * (float)min_val;

        float sub_sum = 0.0f;
        const unsigned char* qp = qs + sub * 16;
        int base_idx = b * 256 + sub * 32;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            unsigned char packed = __ldg(qp + j);
            float w0 = sc_f * (float)(packed & 0x0F) - mn_f;
            float w1 = sc_f * (float)(packed >> 4) - mn_f;
            sub_sum += w0 * s_input[base_idx + j] + w1 * s_input[base_idx + j + 16];
        }
        partial += sub_sum;
    }

    /* Warp-only reduction (32 threads = 1 warp, no block reduction needed) */
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
    if (tid == 0) output[row] = partial;
}

/* Expert-optimized Q5_K kernel — same pattern as Q4_K but with high-bit extraction */
__global__ void q5k_expert_kernel(float* output, const unsigned char* weights,
                                   const float* input, int in_dim, int out_dim) {
    int row = blockIdx.x;
    if (row >= out_dim) return;
    int tid = threadIdx.x;
    int blocks_per_row = in_dim / 256;
    const unsigned char* row_data = weights + (long long)row * blocks_per_row * 176;

    extern __shared__ float s_input[];
    for (int i = tid; i < in_dim; i += EXPERT_THREADS)
        s_input[i] = input[i];
    __syncthreads();

    float partial = 0.0f;
    int total_subs = blocks_per_row * 8;

    for (int si = tid; si < total_subs; si += EXPERT_THREADS) {
        int b = si / 8;
        int sub = si % 8;

        const unsigned char* blk = row_data + b * 176;
        half d_h, dmin_h;
        memcpy(&d_h, blk, 2);
        memcpy(&dmin_h, blk + 2, 2);
        float d = __half2float(d_h);
        float dmin = __half2float(dmin_h);

        const unsigned char* sc = blk + 4;
        const unsigned char* qh = blk + 16;
        const unsigned char* qs = blk + 48;

        int scale_val, min_val;
        gpu_get_scale_min(sub, sc, &scale_val, &min_val);
        float sc_f = d * (float)scale_val;
        float mn_f = dmin * (float)min_val;

        float sub_sum = 0.0f;
        const unsigned char* qp = qs + sub * 16;
        int base_idx = b * 256 + sub * 32;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            unsigned char packed = __ldg(qp + j);
            int wi0 = sub * 32 + j;       /* first 16 weights (blocked) */
            int wi1 = sub * 32 + j + 16; /* second 16 weights */
            int q0 = (packed & 0x0F) | (((qh[wi0 / 8] >> (wi0 % 8)) & 1) << 4);
            int q1 = (packed >> 4) | (((qh[wi1 / 8] >> (wi1 % 8)) & 1) << 4);
            float w0 = sc_f * (float)q0 - mn_f;
            float w1 = sc_f * (float)q1 - mn_f;
            sub_sum += w0 * s_input[base_idx + j] + w1 * s_input[base_idx + j + 16];
        }
        partial += sub_sum;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
    if (tid == 0) output[row] = partial;
}

/* SwiGLU kernel — gate * sigmoid(gate) * up, all on GPU */
__global__ void swiglu_kernel(float* output, const float* gate, const float* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float sig = 0.5f + 0.5f * g / (1.0f + fabsf(g)); /* hard sigmoid — matches CPU */
        output[i] = g * sig * up[i];
    }
}

/* Weighted accumulation: moe_out[i] += weight * expert_out[i] */
__global__ void weighted_add_kernel(float* moe_out, const float* expert_out, float weight, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) moe_out[i] += weight * expert_out[i];
}

/* Zero buffer kernel */
__global__ void zero_kernel(float* buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = 0.0f;
}

/* Forward declarations for io buffers */
static float* d_input_fp32 = NULL;
static float* d_output_fp32 = NULL;
static int d_io_size = 0;

/* Pre-allocated expert FFN buffers (eliminates cudaMalloc/cudaFree per call) */
static float* d_expert_gate = NULL;
static float* d_expert_up = NULL;
static float* d_expert_act = NULL;
static float* d_expert_input = NULL;   /* Dedicated device input for expert stream */
static float* d_expert_output = NULL;  /* Dedicated device output for expert stream */
static float* d_moe_out = NULL;        /* GPU accumulation buffer for batched experts */
static int d_moe_out_size = 0;
static float* h_expert_input = NULL;   /* PINNED host input for truly async H2D */
static float* h_expert_output = NULL;  /* PINNED host output for truly async D2H */
static int d_expert_buf_size = 0;
static int d_expert_io_size = 0;

static int ensure_io_buffers(int max_dim) {
    if (max_dim <= d_io_size) return 0;
    if (d_input_fp32) cudaFree(d_input_fp32);
    if (d_output_fp32) cudaFree(d_output_fp32);
    cudaMalloc(&d_input_fp32, max_dim * sizeof(float));
    cudaMalloc(&d_output_fp32, max_dim * sizeof(float));
    d_io_size = max_dim;
    return 0;
}

static int ensure_expert_buffers(int intermediate) {
    if (intermediate <= d_expert_buf_size) return 0;
    if (d_expert_gate) cudaFree(d_expert_gate);
    if (d_expert_up) cudaFree(d_expert_up);
    if (d_expert_act) cudaFree(d_expert_act);
    cudaMalloc(&d_expert_gate, intermediate * sizeof(float));
    cudaMalloc(&d_expert_up, intermediate * sizeof(float));
    cudaMalloc(&d_expert_act, intermediate * sizeof(float));
    d_expert_buf_size = intermediate;
    return 0;
}

static int ensure_expert_io(int dim) {
    if (dim <= d_expert_io_size) return 0;
    if (d_expert_input) cudaFree(d_expert_input);
    if (d_expert_output) cudaFree(d_expert_output);
    if (h_expert_input) cudaFreeHost(h_expert_input);
    if (h_expert_output) cudaFreeHost(h_expert_output);
    cudaMalloc(&d_expert_input, dim * sizeof(float));
    cudaMalloc(&d_expert_output, dim * sizeof(float));
    cudaMallocHost(&h_expert_input, dim * sizeof(float));   /* Pinned for async */
    cudaMallocHost(&h_expert_output, dim * sizeof(float));  /* Pinned for async */
    d_expert_io_size = dim;
    return 0;
}

/* === GPU Expert Cache === */
#define MAX_GPU_EXPERTS 300
typedef struct {
    void* d_gate;    /* Q4_K gate weights on GPU */
    void* d_up;      /* Q4_K up weights on GPU */
    void* d_down;    /* Q5_K down weights on GPU */
    int layer;
    int expert_id;
    int loaded;
} GPUExpert;

static GPUExpert g_gpu_experts[MAX_GPU_EXPERTS];
static int g_gpu_expert_count = 0;

extern "C" int gpu_cache_expert(int layer, int expert_id,
    const void* gate_data, int gate_size,
    const void* up_data, int up_size,
    const void* down_data, int down_size)
{
    if (g_gpu_expert_count >= MAX_GPU_EXPERTS) return -1;
    GPUExpert* ge = &g_gpu_experts[g_gpu_expert_count];

    CHECK_CUDA(cudaMalloc(&ge->d_gate, gate_size));
    CHECK_CUDA(cudaMemcpy(ge->d_gate, gate_data, gate_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&ge->d_up, up_size));
    CHECK_CUDA(cudaMemcpy(ge->d_up, up_data, up_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&ge->d_down, down_size));
    CHECK_CUDA(cudaMemcpy(ge->d_down, down_data, down_size, cudaMemcpyHostToDevice));

    ge->layer = layer;
    ge->expert_id = expert_id;
    ge->loaded = 1;
    g_gpu_expert_count++;
    g_vram_used += (gate_size + up_size + down_size) / (1024.0f * 1024.0f);
    return g_gpu_expert_count - 1;
}

extern "C" int gpu_find_cached_expert(int layer, int expert_id) {
    for (int i = 0; i < g_gpu_expert_count; i++) {
        if (g_gpu_experts[i].layer == layer && g_gpu_experts[i].expert_id == expert_id)
            return i;
    }
    return -1;
}

extern "C" int gpu_expert_ffn(int cache_idx, const float* input, int hidden_dim,
    int intermediate, float* gate_out, float* up_out, float* expert_out,
    int gate_type, int down_type)
{
    GPUExpert* ge = &g_gpu_experts[cache_idx];
    ensure_io_buffers(hidden_dim > intermediate ? hidden_dim : intermediate);
    ensure_expert_buffers(intermediate);

    /* Upload input once */
    CHECK_CUDA(cudaMemcpy(d_input_fp32, input, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));

    /* Gate matmul — pre-allocated buffer */
    q4k_matvec_kernel<<<intermediate, MATVEC_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
        d_expert_gate, (const unsigned char*)ge->d_gate, d_input_fp32, hidden_dim, intermediate);

    /* Up matmul — pre-allocated buffer */
    q4k_matvec_kernel<<<intermediate, MATVEC_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
        d_expert_up, (const unsigned char*)ge->d_up, d_input_fp32, hidden_dim, intermediate);

    /* Download gate+up for CPU SwiGLU (legacy path) */
    CHECK_CUDA(cudaMemcpy(gate_out, d_expert_gate, intermediate * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(up_out, d_expert_up, intermediate * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int gpu_expert_down(int cache_idx, const float* act, int intermediate,
    float* output, int hidden_dim)
{
    GPUExpert* ge = &g_gpu_experts[cache_idx];
    ensure_io_buffers(hidden_dim > intermediate ? hidden_dim : intermediate);

    CHECK_CUDA(cudaMemcpy(d_input_fp32, act, intermediate * sizeof(float), cudaMemcpyHostToDevice));

    q5k_matvec_kernel<<<hidden_dim, MATVEC_THREADS, intermediate * sizeof(float), g_stream_compute>>>(
        d_output_fp32, (const unsigned char*)ge->d_down, d_input_fp32, intermediate, hidden_dim);

    CHECK_CUDA(cudaMemcpy(output, d_output_fp32, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

/* Fully fused GPU expert: gate+up+swiglu+down all on GPU, single upload/download
 * Runs on g_stream_compute — serialized with DeltaNet gives best throughput
 * (separate streams cause SM contention on single GPU, tested v9.1/v9.2) */
extern "C" int gpu_expert_ffn_fused(int cache_idx, const float* input, int hidden_dim,
    int intermediate, float* expert_out)
{
    GPUExpert* ge = &g_gpu_experts[cache_idx];
    ensure_expert_buffers(intermediate);
    ensure_expert_io(hidden_dim > intermediate ? hidden_dim : intermediate);

    /* Upload input — sync copy to dedicated expert buffer */
    CHECK_CUDA(cudaMemcpy(d_expert_input, input, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));

    /* Gate matmul — expert-optimized kernel (32 threads, full utilization) */
    q4k_expert_kernel<<<intermediate, EXPERT_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
        d_expert_gate, (const unsigned char*)ge->d_gate, d_expert_input, hidden_dim, intermediate);

    /* Up matmul */
    q4k_expert_kernel<<<intermediate, EXPERT_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
        d_expert_up, (const unsigned char*)ge->d_up, d_expert_input, hidden_dim, intermediate);

    /* SwiGLU on GPU */
    int swiglu_blocks = (intermediate + EXPERT_THREADS - 1) / EXPERT_THREADS;
    swiglu_kernel<<<swiglu_blocks, EXPERT_THREADS, 0, g_stream_compute>>>(
        d_expert_act, d_expert_gate, d_expert_up, intermediate);

    /* Down matmul — expert-optimized Q5_K */
    q5k_expert_kernel<<<hidden_dim, EXPERT_THREADS, intermediate * sizeof(float), g_stream_compute>>>(
        d_expert_output, (const unsigned char*)ge->d_down, d_expert_act, intermediate, hidden_dim);

    /* Download output — sync copy waits for all above */
    CHECK_CUDA(cudaMemcpy(expert_out, d_expert_output, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

/* === Batched Expert API — single upload/download per layer === */

/* Start a batched expert block: upload input once, zero moe_out on GPU */
extern "C" int gpu_expert_batch_start(const float* normed, int hidden_dim) {
    ensure_expert_buffers(hidden_dim);  /* reuse intermediate buffers */
    ensure_expert_io(hidden_dim);
    if (hidden_dim > d_moe_out_size) {
        if (d_moe_out) cudaFree(d_moe_out);
        cudaMalloc(&d_moe_out, hidden_dim * sizeof(float));
        d_moe_out_size = hidden_dim;
    }
    /* Upload normed ONCE */
    CHECK_CUDA(cudaMemcpy(d_expert_input, normed, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    /* Zero moe_out accumulator on GPU */
    int zblocks = (hidden_dim + 255) / 256;
    zero_kernel<<<zblocks, 256, 0, g_stream_compute>>>(d_moe_out, hidden_dim);
    return 0;
}

/* Add one expert to the batch — all kernels queued on g_stream_compute, no sync */
extern "C" int gpu_expert_batch_add(int cache_idx, int hidden_dim,
    int intermediate, float weight)
{
    GPUExpert* ge = &g_gpu_experts[cache_idx];
    ensure_expert_buffers(intermediate);

    /* Gate matmul — input already on GPU from batch_start */
    q4k_expert_kernel<<<intermediate, EXPERT_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
        d_expert_gate, (const unsigned char*)ge->d_gate, d_expert_input, hidden_dim, intermediate);

    /* Up matmul */
    q4k_expert_kernel<<<intermediate, EXPERT_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
        d_expert_up, (const unsigned char*)ge->d_up, d_expert_input, hidden_dim, intermediate);

    /* SwiGLU on GPU */
    int swiglu_blocks = (intermediate + EXPERT_THREADS - 1) / EXPERT_THREADS;
    swiglu_kernel<<<swiglu_blocks, EXPERT_THREADS, 0, g_stream_compute>>>(
        d_expert_act, d_expert_gate, d_expert_up, intermediate);

    /* Down matmul */
    q5k_expert_kernel<<<hidden_dim, EXPERT_THREADS, intermediate * sizeof(float), g_stream_compute>>>(
        d_expert_output, (const unsigned char*)ge->d_down, d_expert_act, intermediate, hidden_dim);

    /* Weighted accumulation on GPU — no download needed */
    int ablocks = (hidden_dim + 255) / 256;
    weighted_add_kernel<<<ablocks, 256, 0, g_stream_compute>>>(
        d_moe_out, d_expert_output, weight, hidden_dim);

    return 0;
}

/* Finish batch: download accumulated moe_out */
extern "C" int gpu_expert_batch_finish(float* moe_out, int hidden_dim) {
    CHECK_CUDA(cudaMemcpy(moe_out, d_moe_out, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int gpu_expert_cache_count(void) { return g_gpu_expert_count; }

/* === GPU Standard Attention (GQA) === */

extern "C" int gpu_upload_gqa_weights(int layer,
    const void* wq_q8, int wq_rows, int wq_cols,
    const void* wk_q8, int wk_rows, int wk_cols,
    const void* wv_q8, int wv_rows, int wv_cols,
    const void* wo_q8, int wo_rows, int wo_cols)
{
    if (layer >= MAX_LAYERS) return -1;
    LayerGPU* lg = &g_layers[layer];

    /* Q8_0: (rows * cols / 32) * 34 bytes */
    int wq_size = (wq_rows * wq_cols / 32) * 34;
    int wk_size = (wk_rows * wk_cols / 32) * 34;
    int wv_size = (wv_rows * wv_cols / 32) * 34;
    int wo_size = (wo_rows * wo_cols / 32) * 34;

    CHECK_CUDA(cudaMalloc(&lg->d_wq, wq_size));
    CHECK_CUDA(cudaMemcpy(lg->d_wq, wq_q8, wq_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&lg->d_wk, wk_size));
    CHECK_CUDA(cudaMemcpy(lg->d_wk, wk_q8, wk_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&lg->d_wv, wv_size));
    CHECK_CUDA(cudaMemcpy(lg->d_wv, wv_q8, wv_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&lg->d_wo, wo_size));
    CHECK_CUDA(cudaMemcpy(lg->d_wo, wo_q8, wo_size, cudaMemcpyHostToDevice));

    lg->wq_rows = wq_rows; lg->wq_cols = wq_cols;
    lg->wk_rows = wk_rows; lg->wk_cols = wk_cols;
    lg->wv_rows = wv_rows; lg->wv_cols = wv_cols;
    lg->wo_rows = wo_rows; lg->wo_cols = wo_cols;
    lg->gqa_loaded = 1;

    g_vram_used += (wq_size + wk_size + wv_size + wo_size) / (1024.0f * 1024.0f);
    return 0;
}

/* GPU GQA projections: Q+gate, K, V all on GPU, download results */
extern "C" int gpu_gqa_projections(int layer,
    const float* normed, int hidden_dim,
    float* q_gate_out, int q_gate_dim,
    float* k_out, int k_dim,
    float* v_out, int v_dim)
{
    LayerGPU* lg = &g_layers[layer];
    if (!lg->gqa_loaded) return -1;
    ensure_io_buffers(q_gate_dim > hidden_dim ? q_gate_dim : hidden_dim);

    /* Upload normed input */
    CHECK_CUDA(cudaMemcpy(d_input_fp32, normed, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));

    /* Q+Gate projection: [hidden → q_gate_dim] */
    q8_matvec_kernel<<<q_gate_dim, MATVEC_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
        d_output_fp32, (const unsigned char*)lg->d_wq, d_input_fp32, hidden_dim, q_gate_dim);
    CHECK_CUDA(cudaMemcpy(q_gate_out, d_output_fp32, q_gate_dim * sizeof(float), cudaMemcpyDeviceToHost));

    /* K projection: [hidden → k_dim] */
    q8_matvec_kernel<<<k_dim, MATVEC_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
        d_output_fp32, (const unsigned char*)lg->d_wk, d_input_fp32, hidden_dim, k_dim);
    CHECK_CUDA(cudaMemcpy(k_out, d_output_fp32, k_dim * sizeof(float), cudaMemcpyDeviceToHost));

    /* V projection: [hidden → v_dim] */
    q8_matvec_kernel<<<v_dim, MATVEC_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
        d_output_fp32, (const unsigned char*)lg->d_wv, d_input_fp32, hidden_dim, v_dim);
    CHECK_CUDA(cudaMemcpy(v_out, d_output_fp32, v_dim * sizeof(float), cudaMemcpyDeviceToHost));

    return 0;
}

/* GPU O projection: [attn_dim → hidden_dim] */
extern "C" int gpu_gqa_output(int layer,
    const float* attn_out, int attn_dim,
    float* output, int hidden_dim)
{
    LayerGPU* lg = &g_layers[layer];
    if (!lg->gqa_loaded) return -1;
    ensure_io_buffers(attn_dim > hidden_dim ? attn_dim : hidden_dim);

    CHECK_CUDA(cudaMemcpy(d_input_fp32, attn_out, attn_dim * sizeof(float), cudaMemcpyHostToDevice));
    q8_matvec_kernel<<<hidden_dim, MATVEC_THREADS, attn_dim * sizeof(float), g_stream_compute>>>(
        d_output_fp32, (const unsigned char*)lg->d_wo, d_input_fp32, attn_dim, hidden_dim);
    CHECK_CUDA(cudaMemcpy(output, d_output_fp32, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

/* === GPU Router === */
static float* d_router_weights[MAX_LAYERS] = {0};
static int d_router_loaded[MAX_LAYERS] = {0};

extern "C" int gpu_upload_router(int layer, const float* weights, int hidden_dim, int num_experts) {
    if (layer >= MAX_LAYERS) return -1;
    int size = hidden_dim * num_experts * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_router_weights[layer], size));
    CHECK_CUDA(cudaMemcpy(d_router_weights[layer], weights, size, cudaMemcpyHostToDevice));
    d_router_loaded[layer] = 1;
    g_vram_used += size / (1024.0f * 1024.0f);
    return 0;
}

/* FP32 matvec kernel for router (same pattern as Q8_0 but simpler — no dequant) */
__global__ void fp32_matvec_kernel(float* output, const float* weights,
                                    const float* input, int in_dim, int out_dim) {
    int row = blockIdx.x;
    if (row >= out_dim) return;
    int tid = threadIdx.x;
    const float* row_data = weights + (long long)row * in_dim;

    float partial = 0.0f;
    for (int j = tid; j < in_dim; j += MATVEC_THREADS)
        partial += row_data[j] * input[j];

    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
    __shared__ float shared[8];
    int warp_id = tid / 32, lane_id = tid % 32;
    if (lane_id == 0) shared[warp_id] = partial;
    __syncthreads();
    if (tid < 8) {
        float val = shared[tid];
        for (int offset = 4; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFF, val, offset);
        if (tid == 0) output[row] = val;
    }
}

extern "C" int gpu_router(int layer, const float* normed, int hidden_dim,
                           float* logits_out, int num_experts) {
    if (!d_router_loaded[layer]) return -1;
    ensure_io_buffers(hidden_dim > num_experts ? hidden_dim : num_experts);

    CHECK_CUDA(cudaMemcpy(d_input_fp32, normed, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    fp32_matvec_kernel<<<num_experts, MATVEC_THREADS, 0, g_stream_compute>>>(
        d_output_fp32, d_router_weights[layer], d_input_fp32, hidden_dim, num_experts);
    CHECK_CUDA(cudaMemcpy(logits_out, d_output_fp32, num_experts * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int gpu_init(void) {
    if (g_initialized) return 0;
    CHECK_CUDA(cudaSetDevice(0));
    cublasCreate(&g_cublas);
    memset(g_layers, 0, sizeof(g_layers));

    /* Create 3 non-blocking streams */
    CHECK_CUDA(cudaStreamCreateWithFlags(&g_stream_compute, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&g_stream_h2d, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&g_stream_d2h, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&g_stream_expert, cudaStreamNonBlocking));

    /* Allocate double-buffered pipeline slots */
    for (int s = 0; s < NUM_PIPELINE_SLOTS; s++) {
        PipelineSlot* slot = &g_slots[s];
        CHECK_CUDA(cudaMallocHost(&slot->h_normed, 8192 * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&slot->h_qkv_out, 16384 * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&slot->h_gate_out, 8192 * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&slot->h_gated_in, 8192 * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&slot->h_ssm_out, 8192 * sizeof(float)));

        CHECK_CUDA(cudaMalloc(&slot->d_normed, 8192 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&slot->d_qkv_out, 16384 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&slot->d_gate_out, 8192 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&slot->d_gated_in, 8192 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&slot->d_ssm_out, 8192 * sizeof(float)));

        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_h2d_normed_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_qkv_gate_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_d2h_qkv_gate_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_h2d_gated_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_ssm_out_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_d2h_ssm_out_done, cudaEventDisableTiming));
    }

    g_initialized = 1;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "GPU initialized: %s, %.0f MB VRAM, compute %d.%d, 4 streams\n",
            prop.name, prop.totalGlobalMem / (1024.0f * 1024.0f), prop.major, prop.minor);
    fflush(stderr);
    return 0;
}

extern "C" int gpu_upload_deltanet_weights(int layer,
    const void* qkv_q8, int qkv_rows, int qkv_cols,
    const void* gate_q8, int gate_rows, int gate_cols,
    const void* ssm_out_q8, int ssm_rows, int ssm_cols)
{
    if (layer >= MAX_LAYERS) return -1;
    LayerGPU* lg = &g_layers[layer];
    lg->qkv_rows = qkv_rows; lg->qkv_cols = qkv_cols;
    lg->gate_rows = gate_rows; lg->gate_cols = gate_cols;
    lg->ssm_rows = ssm_rows; lg->ssm_cols = ssm_cols;

    int qkv_q8_size = (qkv_rows * qkv_cols / 32) * 34;
    int gate_q8_size = (gate_rows * gate_cols / 32) * 34;
    int ssm_q8_size = (ssm_rows * ssm_cols / 32) * 34;

    CHECK_CUDA(cudaMalloc(&lg->d_qkv, qkv_q8_size));
    CHECK_CUDA(cudaMemcpy(lg->d_qkv, qkv_q8, qkv_q8_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&lg->d_gate, gate_q8_size));
    CHECK_CUDA(cudaMemcpy(lg->d_gate, gate_q8, gate_q8_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&lg->d_ssm_out, ssm_q8_size));
    CHECK_CUDA(cudaMemcpy(lg->d_ssm_out, ssm_out_q8, ssm_q8_size, cudaMemcpyHostToDevice));

    g_vram_used += (qkv_q8_size + gate_q8_size + ssm_q8_size) / (1024.0f * 1024.0f);
    lg->loaded = 1;

    g_hidden_dim = qkv_rows;
    g_qkv_dim = qkv_cols;
    g_gate_dim = gate_cols;
    return 0;
}

/* === ASYNC API (Phase 3 Pipeline) === */

extern "C" int gpu_launch_qkv_gate(int layer, int slot_idx,
    const float* normed_src, int hidden_dim, int qkv_dim, int gate_dim)
{
    LayerGPU* lg = &g_layers[layer];
    if (!lg->loaded) return -1;
    PipelineSlot* slot = &g_slots[slot_idx];

    /* Copy normed into pinned buffer */
    memcpy(slot->h_normed, normed_src, hidden_dim * sizeof(float));

    /* Async H2D */
    cudaMemcpyAsync(slot->d_normed, slot->h_normed,
        hidden_dim * sizeof(float), cudaMemcpyHostToDevice, g_stream_h2d);
    cudaEventRecord(slot->evt_h2d_normed_done, g_stream_h2d);

    /* Compute waits for H2D */
    cudaStreamWaitEvent(g_stream_compute, slot->evt_h2d_normed_done, 0);

    /* Launch QKV + Gate kernels */
    q8_matvec_kernel<<<qkv_dim, MATVEC_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
        slot->d_qkv_out, (const unsigned char*)lg->d_qkv,
        slot->d_normed, hidden_dim, qkv_dim);
    q8_matvec_kernel<<<gate_dim, MATVEC_THREADS, hidden_dim * sizeof(float), g_stream_compute>>>(
        slot->d_gate_out, (const unsigned char*)lg->d_gate,
        slot->d_normed, hidden_dim, gate_dim);
    cudaEventRecord(slot->evt_qkv_gate_done, g_stream_compute);

    /* D2H waits for compute, copies results */
    cudaStreamWaitEvent(g_stream_d2h, slot->evt_qkv_gate_done, 0);
    cudaMemcpyAsync(slot->h_qkv_out, slot->d_qkv_out,
        qkv_dim * sizeof(float), cudaMemcpyDeviceToHost, g_stream_d2h);
    cudaMemcpyAsync(slot->h_gate_out, slot->d_gate_out,
        gate_dim * sizeof(float), cudaMemcpyDeviceToHost, g_stream_d2h);
    cudaEventRecord(slot->evt_d2h_qkv_gate_done, g_stream_d2h);

    return 0; /* Returns immediately */
}

extern "C" int gpu_wait_qkv_gate(int slot_idx) {
    cudaEventSynchronize(g_slots[slot_idx].evt_d2h_qkv_gate_done);
    return 0;
}

extern "C" float* gpu_get_qkv_out(int slot_idx) { return g_slots[slot_idx].h_qkv_out; }
extern "C" float* gpu_get_gate_out(int slot_idx) { return g_slots[slot_idx].h_gate_out; }
extern "C" float* gpu_get_ssm_out_buf(int slot_idx) { return g_slots[slot_idx].h_ssm_out; }

extern "C" int gpu_launch_ssm_out(int layer, int slot_idx,
    const float* gated_src, int gated_dim, int output_dim)
{
    LayerGPU* lg = &g_layers[layer];
    if (!lg->loaded) return -1;
    PipelineSlot* slot = &g_slots[slot_idx];

    memcpy(slot->h_gated_in, gated_src, gated_dim * sizeof(float));

    cudaMemcpyAsync(slot->d_gated_in, slot->h_gated_in,
        gated_dim * sizeof(float), cudaMemcpyHostToDevice, g_stream_h2d);
    cudaEventRecord(slot->evt_h2d_gated_done, g_stream_h2d);

    cudaStreamWaitEvent(g_stream_compute, slot->evt_h2d_gated_done, 0);
    q8_matvec_kernel<<<output_dim, MATVEC_THREADS, gated_dim * sizeof(float), g_stream_compute>>>(
        slot->d_ssm_out, (const unsigned char*)lg->d_ssm_out,
        slot->d_gated_in, gated_dim, output_dim);
    cudaEventRecord(slot->evt_ssm_out_done, g_stream_compute);

    cudaStreamWaitEvent(g_stream_d2h, slot->evt_ssm_out_done, 0);
    cudaMemcpyAsync(slot->h_ssm_out, slot->d_ssm_out,
        output_dim * sizeof(float), cudaMemcpyDeviceToHost, g_stream_d2h);
    cudaEventRecord(slot->evt_d2h_ssm_out_done, g_stream_d2h);

    return 0;
}

extern "C" int gpu_wait_ssm_out(int slot_idx) {
    cudaEventSynchronize(g_slots[slot_idx].evt_d2h_ssm_out_done);
    return 0;
}

/* Legacy sync API (calls async then waits — for backward compat) */
extern "C" int gpu_deltanet_projections(int layer,
    const float* normed, int hidden_dim,
    float* qkv_out, int qkv_dim,
    float* gate_out, int gate_dim)
{
    int slot = 0;
    gpu_launch_qkv_gate(layer, slot, normed, hidden_dim, qkv_dim, gate_dim);
    gpu_wait_qkv_gate(slot);
    memcpy(qkv_out, g_slots[slot].h_qkv_out, qkv_dim * sizeof(float));
    memcpy(gate_out, g_slots[slot].h_gate_out, gate_dim * sizeof(float));
    return 0;
}

extern "C" int gpu_ssm_out_projection(int layer,
    const float* gated, int gated_dim,
    float* output, int output_dim)
{
    int slot = 0;
    gpu_launch_ssm_out(layer, slot, gated, gated_dim, output_dim);
    gpu_wait_ssm_out(slot);
    memcpy(output, g_slots[slot].h_ssm_out, output_dim * sizeof(float));
    return 0;
}

extern "C" void gpu_shutdown(void) {
    if (!g_initialized) return;
    cudaDeviceSynchronize();
    for (int i = 0; i < MAX_LAYERS; i++) {
        if (g_layers[i].loaded) {
            cudaFree(g_layers[i].d_qkv);
            cudaFree(g_layers[i].d_gate);
            cudaFree(g_layers[i].d_ssm_out);
        }
    }
    for (int s = 0; s < NUM_PIPELINE_SLOTS; s++) {
        PipelineSlot* slot = &g_slots[s];
        cudaFreeHost(slot->h_normed); cudaFreeHost(slot->h_qkv_out);
        cudaFreeHost(slot->h_gate_out); cudaFreeHost(slot->h_gated_in);
        cudaFreeHost(slot->h_ssm_out);
        cudaFree(slot->d_normed); cudaFree(slot->d_qkv_out);
        cudaFree(slot->d_gate_out); cudaFree(slot->d_gated_in);
        cudaFree(slot->d_ssm_out);
        cudaEventDestroy(slot->evt_h2d_normed_done);
        cudaEventDestroy(slot->evt_qkv_gate_done);
        cudaEventDestroy(slot->evt_d2h_qkv_gate_done);
        cudaEventDestroy(slot->evt_h2d_gated_done);
        cudaEventDestroy(slot->evt_ssm_out_done);
        cudaEventDestroy(slot->evt_d2h_ssm_out_done);
    }
    if (g_stream_compute) cudaStreamDestroy(g_stream_compute);
    if (g_stream_h2d) cudaStreamDestroy(g_stream_h2d);
    if (g_stream_d2h) cudaStreamDestroy(g_stream_d2h);
    if (g_stream_expert) cudaStreamDestroy(g_stream_expert);
    if (g_cublas) cublasDestroy(g_cublas);
    g_initialized = 0;
}

extern "C" int gpu_is_initialized(void) { return g_initialized; }
extern "C" float gpu_vram_used_mb(void) { return g_vram_used; }
