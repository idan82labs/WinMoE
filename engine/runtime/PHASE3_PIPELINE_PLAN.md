# Phase 3: GPU-CPU Pipeline Overlap — Implementation Plan

**Target**: 384ms/token -> ~210ms/token (2.60 tok/s -> ~4.8 tok/s)
**Author**: Generated from code analysis of gpu_offload.cu, winmoe_inference.c, deltanet_impl.h
**Date**: 2026-03-27

---

## Table of Contents

1. [Dependency Analysis](#1-dependency-analysis)
2. [CUDA Stream Topology](#2-cuda-stream-topology)
3. [Pinned Memory Layout](#3-pinned-memory-layout)
4. [Synchronization Events](#4-synchronization-events)
5. [The Pipelined Layer Loop State Machine](#5-the-pipelined-layer-loop-state-machine)
6. [Double-Buffering Design](#6-double-buffering-design)
7. [Standard Attention Layers (Every 4th)](#7-standard-attention-layers-every-4th)
8. [Expert FFN Overlap with GPU](#8-expert-ffn-overlap-with-gpu)
9. [Error Handling and Fault Tolerance](#9-error-handling-and-fault-tolerance)
10. [Race Conditions and Deadlock Analysis](#10-race-conditions-and-deadlock-analysis)
11. [VRAM Budget](#11-vram-budget)
12. [Implementation Phases](#12-implementation-phases)
13. [Expected Performance Model](#13-expected-performance-model)

---

## 1. Dependency Analysis

### Current Sequential Flow Per DeltaNet Layer L

```
Step  What                          Where    Input From          Output To         Time
----  ----                          -----    ----------          ---------         ----
A     RMSNorm(hidden)               CPU      hidden[L]           normed[L]         0.05ms
B     QKV projection [4096->8192]   GPU      normed[L]           qkv[L]            0.8ms
C     Gate projection [4096->4096]  GPU      normed[L]           gate[L]           0.8ms
D     Alpha+Beta projections        CPU      normed[L]           alpha[L],beta[L]  0.05ms
E     State recurrence (32 heads)   CPU      qkv[L],alpha,beta   head_out[L]       1.0ms
F     RMSNorm + SiLU gating         CPU      head_out[L],gate[L] gated[L]          0.1ms
G     SSM Out projection [4096->4096] GPU    gated[L]            ssm_out[L]        0.4ms
H     Residual add                  CPU      ssm_out[L],residual hidden[L+1]       0.01ms
I     Post-attention norm           CPU      hidden[L+1]         normed_post[L]    0.05ms
J     Router topK                   CPU      normed_post[L]      expert_ids[L]     0.4ms
K     Expert FFN (K=4 experts)      CPU      normed_post[L]      moe_out[L]        2.5ms
M     Residual add                  CPU      moe_out[L],residual hidden_final[L]   0.01ms
```

### True Data Dependencies (Cannot Be Broken)

```
A -> B,C,D    (normed needed by projections)
B -> E        (QKV needed by state recurrence) -- GPU->CPU dependency
C -> F        (gate needed by SiLU gating) -- GPU->CPU dependency
D -> E        (alpha/beta needed by state recurrence)
E -> F        (head_out needed by gating)
F -> G        (gated needed by SSM Out) -- CPU->GPU dependency
G -> H        (ssm_out needed by residual) -- GPU->CPU dependency
H -> I        (hidden needed by post-norm)
I -> J,K      (normed_post needed by router/experts)
K -> M        (moe_out needed by residual)
M -> A(L+1)   (hidden_final is input to next layer)
```

### The Key Insight: What CAN Overlap

Within a single layer, steps B+C (GPU, 1.6ms) overlap with step D (CPU, 0.05ms).
That is the only intra-layer overlap because every subsequent step depends on GPU output.

**Inter-layer overlap is where the win lives:**

```
Layer L steps K (Expert FFN, 2.5ms) can overlap with Layer L+1 steps B+C (GPU, 1.6ms)
```

But there is a critical ordering constraint:
- Layer L+1 step A (RMSNorm) needs `hidden_final[L]` which needs `moe_out[L]` from step K.
- So we CANNOT start Layer L+1 GPU work until Layer L Expert FFN finishes.

**Wait. Re-examining.** The residual from attention (step H) is computed BEFORE expert FFN starts.
The post-attention `hidden[L]` (after step H) feeds into:
1. Post-attention norm (step I) -> Router (step J) -> Expert FFN (step K)
2. Expert output adds back to residual from step H's `hidden[L]`, producing `hidden_final[L]`

So `hidden_final[L] = hidden_after_attn[L] + moe_out[L]`

And Layer L+1 needs `hidden_final[L]` for its RMSNorm. This means we truly cannot start
Layer L+1 until Layer L's Expert FFN completes.

### Revised Overlap Strategy: Intra-DeltaNet Pipelining + Async GPU

The real overlap opportunities are:

1. **GPU async fire-and-forget**: Launch GPU kernels asynchronously, do CPU work while waiting
2. **Alpha/Beta parallel with QKV/Gate on GPU**: D overlaps B+C (saves 0.05ms, minor)
3. **Expert I/O prefetch**: Start SSD reads for Layer L+1 experts while computing Layer L experts
4. **SSM Out projection overlap with post-processing**: Launch SSM Out on GPU, do norm on CPU

But the BIGGEST win comes from restructuring to **overlap Layer L's SSM Out GPU work
with Layer L's CPU post-processing tail**, and **overlapping expert SSD I/O with compute**.

### Revised Realistic Overlap Map

```
Time ->
GPU:   [QKV+Gate L (1.6ms)   ][idle 1.1ms ][SSM_Out L (0.4ms)][QKV+Gate L+1 ...
CPU:   [Alpha/Beta(0.05)][wait][Recur(1.0)][Gating(0.1)][wait ][Resid+Norm+Route+Experts(3.0ms)]

PIPELINED:
GPU:   [QKV+Gate L (1.6ms)             ][SSM_Out L (0.4ms)][QKV+Gate L+1 (1.6ms)          ]...
CPU:   [Alpha/Beta L (0.05)][===wait===][Recur L (1.0)    ][Gate+Norm(0.1)][===wait(0.1)===][Resid+Route+Experts L (2.96ms)][Alpha/Beta L+1]...
```

The problem is that CPU must wait for GPU QKV result before state recurrence.
And GPU must wait for CPU gated result before SSM Out.

So the intra-layer pipeline has this irreducible serial chain:
```
GPU QKV+Gate: 1.6ms -> CPU Recurrence+Gating: 1.1ms -> GPU SSM_Out: 0.4ms -> CPU Experts: 3.0ms
Total serial: 6.1ms per layer (worse than current because of sync overhead!)
```

### THE ACTUAL WIN: Overlapping Expert FFN with NEXT LAYER'S GPU work

After SSM Out completes for Layer L, the CPU computes:
- Residual add (0.01ms)
- Post-attention norm (0.05ms)
- Router topK (0.4ms)
- Expert FFN (2.5ms)
- Residual add (0.01ms)
- RMSNorm for Layer L+1 (0.05ms)

Total CPU-only tail: ~3.0ms

Meanwhile, the GPU is idle after SSM Out L. We CANNOT start QKV+Gate for L+1
until the expert FFN for L finishes and produces `hidden_final[L]`.

**Unless we restructure the computation order.**

### THE BREAKTHROUGH: Speculative RMSNorm / Deferred Residual

Current: `hidden_final[L] = (residual + attn_out) + moe_out`
Then: `normed[L+1] = RMSNorm(hidden_final[L])`

We can rewrite as:
```
hidden_after_attn[L] = residual + attn_out    (known after step H)
hidden_final[L] = hidden_after_attn[L] + moe_out[L]    (known after step K)
```

The RMSNorm for Layer L+1 depends on `hidden_final[L]`, which depends on `moe_out[L]`.
There is no way around this without speculative execution or reformulating the math.

**However**, there IS a more subtle overlap possible:

### REVISED PLAN: 3-Phase Per-Layer Pipeline

**Phase 1: GPU DeltaNet projections (async)**
- CPU launches QKV+Gate kernels on GPU stream
- CPU immediately does Alpha/Beta projections (tiny, overlaps with GPU)
- CPU waits for GPU QKV+Gate result via CUDA event

**Phase 2: CPU DeltaNet state recurrence + GPU SSM Out (partially async)**
- CPU computes state recurrence using QKV (1.0ms)
- CPU computes RMSNorm + SiLU gating (0.1ms)
- CPU launches SSM Out projection on GPU (async)
- CPU starts post-attention norm while GPU computes SSM Out

**Phase 3: CPU Expert FFN (with async SSD I/O prefetch for next layer)**
- CPU waits for SSM Out result, does residual add
- CPU does router topK
- CPU does expert FFN
- WHILE expert FFN runs, prefetch next-layer expert weights from SSD
- CPU does final residual add

The key optimization is making **every GPU launch async** and doing useful CPU work
in every gap, plus **prefetching SSD I/O for expert weights**.

---

## 2. CUDA Stream Topology

### Stream Design: 3 Streams

```
Stream 0 (COMPUTE): QKV projection, Gate projection, SSM Out projection
Stream 1 (H2D):     Host-to-Device async transfers (normed -> GPU, gated -> GPU)
Stream 2 (D2H):     Device-to-Host async transfers (qkv result -> CPU, gate result -> CPU, ssm_out -> CPU)
```

**Why 3 streams, not 2:**
- On RTX 3070 (Ampere), there is 1 copy engine for H2D and 1 for D2H, plus compute.
- Using separate H2D and D2H streams allows the DMA engines to operate independently.
- All compute goes on stream 0 to avoid kernel serialization issues.

**Why not more streams:**
- There is only ONE compute pipeline on consumer GPUs. Multiple compute streams
  would just serialize on the same SM anyway for our matvec kernels (they saturate
  the GPU's memory bandwidth).
- Our kernels are memory-bound (Q8_0 dequant + matvec). Two kernels on separate
  streams would contend for memory bandwidth and be slower than sequential.

### Stream Creation Code

```c
// In gpu_offload.cu — add to global state
static cudaStream_t g_stream_compute = NULL;
static cudaStream_t g_stream_h2d = NULL;
static cudaStream_t g_stream_d2h = NULL;

// In gpu_init():
CHECK_CUDA(cudaStreamCreateWithFlags(&g_stream_compute, cudaStreamNonBlocking));
CHECK_CUDA(cudaStreamCreateWithFlags(&g_stream_h2d, cudaStreamNonBlocking));
CHECK_CUDA(cudaStreamCreateWithFlags(&g_stream_d2h, cudaStreamNonBlocking));

// Bind cuBLAS to compute stream (if we ever use it)
CHECK_CUBLAS(cublasSetStream(g_cublas, g_stream_compute));
```

**CRITICAL: cudaStreamNonBlocking**
Without this flag, streams synchronize implicitly with the default stream (stream 0).
Since our current code uses the default stream (implicit NULL stream), we MUST use
NonBlocking to prevent legacy default-stream synchronization semantics from
serializing everything.

### Stream Shutdown

```c
// In gpu_shutdown():
if (g_stream_compute) cudaStreamDestroy(g_stream_compute);
if (g_stream_h2d) cudaStreamDestroy(g_stream_h2d);
if (g_stream_d2h) cudaStreamDestroy(g_stream_d2h);
```

---

## 3. Pinned Memory Layout

### Current Problem
The current code has ONE set of pinned buffers (`h_input`, `h_output`) and ONE set of
device buffers (`d_input_fp32`, `d_output_fp32`). This makes it impossible to overlap
transfers with compute — you would overwrite the buffer being used by an in-flight kernel.

### New Pinned Buffer Pool: Double-Buffered

We need double-buffered pinned memory for CPU<->GPU transfers. Each "slot" contains
all the buffers needed for one layer's DeltaNet computation.

```c
#define NUM_PIPELINE_SLOTS 2  // Double buffer: slot A and slot B

typedef struct {
    // Pinned host memory (for async H2D/D2H)
    float* h_normed;       // [4096] Input to QKV+Gate projections
    float* h_qkv_out;     // [8192] QKV projection output
    float* h_gate_out;    // [4096] Gate projection output
    float* h_gated_in;    // [4096] Input to SSM Out projection
    float* h_ssm_out;     // [4096] SSM Out projection result

    // Device memory (GPU-side buffers)
    float* d_normed;       // [4096] FP32 input on device
    float* d_qkv_out;     // [8192] QKV result on device
    float* d_gate_out;    // [4096] Gate result on device
    float* d_gated_in;    // [4096] Gated input on device
    float* d_ssm_out;     // [4096] SSM Out result on device

    // CUDA events for synchronization
    cudaEvent_t evt_h2d_normed_done;    // normed uploaded to GPU
    cudaEvent_t evt_qkv_gate_done;      // QKV+Gate kernels finished
    cudaEvent_t evt_d2h_qkv_gate_done;  // QKV+Gate results on CPU
    cudaEvent_t evt_h2d_gated_done;     // gated uploaded to GPU
    cudaEvent_t evt_ssm_out_done;       // SSM Out kernel finished
    cudaEvent_t evt_d2h_ssm_out_done;   // SSM Out result on CPU
} PipelineSlot;

static PipelineSlot g_slots[NUM_PIPELINE_SLOTS];
```

### Memory Sizes and Allocation

```c
// Constants for Qwen3.5-397B
#define HIDDEN_DIM 4096
#define QKV_DIM 8192       // DN_QKV_DIM
#define GATE_DIM 4096      // DN_GATE_DIM = DN_INNER
#define SSM_OUT_DIM 4096   // same as hidden

static int pipeline_slots_init(void) {
    for (int s = 0; s < NUM_PIPELINE_SLOTS; s++) {
        PipelineSlot* slot = &g_slots[s];

        // Pinned host memory (page-locked for async DMA)
        CHECK_CUDA(cudaMallocHost(&slot->h_normed,   HIDDEN_DIM * sizeof(float)));  // 16 KB
        CHECK_CUDA(cudaMallocHost(&slot->h_qkv_out,  QKV_DIM * sizeof(float)));     // 32 KB
        CHECK_CUDA(cudaMallocHost(&slot->h_gate_out,  GATE_DIM * sizeof(float)));    // 16 KB
        CHECK_CUDA(cudaMallocHost(&slot->h_gated_in, GATE_DIM * sizeof(float)));     // 16 KB
        CHECK_CUDA(cudaMallocHost(&slot->h_ssm_out,  SSM_OUT_DIM * sizeof(float))); // 16 KB

        // Device memory
        CHECK_CUDA(cudaMalloc(&slot->d_normed,   HIDDEN_DIM * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&slot->d_qkv_out,  QKV_DIM * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&slot->d_gate_out,  GATE_DIM * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&slot->d_gated_in, GATE_DIM * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&slot->d_ssm_out,  SSM_OUT_DIM * sizeof(float)));

        // Events (disable timing for lower overhead — we profile separately)
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_h2d_normed_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_qkv_gate_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_d2h_qkv_gate_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_h2d_gated_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_ssm_out_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_d2h_ssm_out_done, cudaEventDisableTiming));
    }
    return 0;
}
```

### Total New VRAM Cost

Per slot:
- d_normed: 4096 * 4 = 16 KB
- d_qkv_out: 8192 * 4 = 32 KB
- d_gate_out: 4096 * 4 = 16 KB
- d_gated_in: 4096 * 4 = 16 KB
- d_ssm_out: 4096 * 4 = 16 KB
- Total per slot: 96 KB

2 slots: **192 KB**. Negligible vs. the 6.1 GB already used.

### Total New Pinned Host Memory Cost

Per slot: 5 buffers * 16-32 KB = ~96 KB
2 slots: **192 KB** pinned. Well within system limits.

---

## 4. Synchronization Events

### Event Semantics

CUDA events are recorded into a stream. `cudaEventRecord(event, stream)` places a marker
in the stream's command queue. `cudaStreamWaitEvent(stream, event, 0)` makes `stream`
block until `event` is reached. `cudaEventSynchronize(event)` makes the **CPU** block.

### Event Map for One Layer

```
Timeline (one DeltaNet layer L, using slot S = L % 2):

CPU thread                          GPU stream_h2d         GPU stream_compute      GPU stream_d2h
──────────                          ──────────────         ──────────────────      ──────────────
1. memcpy normed -> h_normed[S]
2. cudaMemcpyAsync(d_normed[S],     [H2D: normed upload]
   h_normed[S], stream_h2d)
3. cudaEventRecord(                  [record evt_h2d_
   evt_h2d_normed_done, stream_h2d)   normed_done]

4. cudaStreamWaitEvent(                                    [wait evt_h2d_normed_done]
   stream_compute, evt_h2d_normed_done)

5. Launch QKV kernel on stream_compute                     [QKV matvec 0.8ms]
6. Launch Gate kernel on stream_compute                    [Gate matvec 0.8ms]
7. cudaEventRecord(                                        [record evt_qkv_gate_done]
   evt_qkv_gate_done, stream_compute)

   === CPU does Alpha/Beta projections here (0.05ms) ===
   === CPU is FREE during GPU QKV+Gate (1.6ms total)  ===

8. cudaStreamWaitEvent(                                                             [wait evt_qkv_gate_done]
   stream_d2h, evt_qkv_gate_done)
9. cudaMemcpyAsync(h_qkv_out[S],                                                   [D2H: QKV 32KB]
   d_qkv_out[S], stream_d2h)
10.cudaMemcpyAsync(h_gate_out[S],                                                   [D2H: Gate 16KB]
   d_gate_out[S], stream_d2h)
11.cudaEventRecord(                                                                  [record evt_d2h_qkv_gate_done]
   evt_d2h_qkv_gate_done, stream_d2h)

12.cudaEventSynchronize(             *** CPU BLOCKS HERE until QKV+Gate on host ***
   evt_d2h_qkv_gate_done)

   === CPU state recurrence (1.0ms) ===
   === CPU RMSNorm + SiLU gating (0.1ms) ===

13.memcpy gated -> h_gated_in[S]
14.cudaMemcpyAsync(d_gated_in[S],    [H2D: gated upload]
   h_gated_in[S], stream_h2d)
15.cudaEventRecord(                   [record evt_h2d_gated_done]
   evt_h2d_gated_done, stream_h2d)

16.cudaStreamWaitEvent(                                    [wait evt_h2d_gated_done]
   stream_compute, evt_h2d_gated_done)
17.Launch SSM_Out kernel on                                [SSM_Out matvec 0.4ms]
   stream_compute
18.cudaEventRecord(                                        [record evt_ssm_out_done]
   evt_ssm_out_done, stream_compute)

19.cudaStreamWaitEvent(                                                             [wait evt_ssm_out_done]
   stream_d2h, evt_ssm_out_done)
20.cudaMemcpyAsync(h_ssm_out[S],                                                   [D2H: SSM_Out 16KB]
   d_ssm_out_out[S], stream_d2h)
21.cudaEventRecord(                                                                  [record evt_d2h_ssm_out_done]
   evt_d2h_ssm_out_done, stream_d2h)

22.cudaEventSynchronize(             *** CPU BLOCKS until SSM_Out on host ***
   evt_d2h_ssm_out_done)

   === CPU residual add (0.01ms) ===
   === CPU post-norm + router + expert FFN (3.0ms) ===
   === CPU final residual add (0.01ms) ===
```

### Where CPU Waits for GPU (2 sync points per DeltaNet layer)

| Sync Point | Event | CPU Blocked Until | Expected Wait |
|-----------|-------|-------------------|---------------|
| **Sync 1** (step 12) | `evt_d2h_qkv_gate_done` | QKV+Gate results are in pinned host memory | ~1.55ms (1.6ms GPU - 0.05ms CPU alpha/beta) |
| **Sync 2** (step 22) | `evt_d2h_ssm_out_done` | SSM Out result is in pinned host memory | ~0ms (CPU gating takes 0.1ms, GPU SSM_Out takes 0.4ms, but CPU finishes first so waits ~0.3ms... but transfer is only 16KB so ~0ms after kernel) |

### Where GPU Waits for CPU (0 explicit waits, but implicit via H2D)

GPU never explicitly waits for CPU. Instead, the GPU command queue is structured so that:
- The H2D transfer of `normed` happens before compute starts (via stream_compute waiting on evt_h2d_normed_done)
- The H2D transfer of `gated` happens before SSM_Out starts (via stream_compute waiting on evt_h2d_gated_done)

The CPU enqueues all GPU work for a layer before blocking. The GPU processes its
queue autonomously. The CPU only blocks when it needs a result.

---

## 5. The Pipelined Layer Loop State Machine

### New GPU API Functions

```c
// gpu_offload.h — new async API

// Launch QKV+Gate projections asynchronously. Results will appear in
// the pinned host buffers of the specified pipeline slot.
// Returns immediately. Call gpu_wait_qkv_gate() before reading results.
int gpu_launch_qkv_gate(int layer, int slot,
    const float* normed_pinned, int hidden_dim,
    int qkv_dim, int gate_dim);

// Block CPU until QKV+Gate results are in pinned host memory.
// After this returns, slot->h_qkv_out and slot->h_gate_out are valid.
int gpu_wait_qkv_gate(int slot);

// Launch SSM Out projection asynchronously.
// Returns immediately. Call gpu_wait_ssm_out() before reading result.
int gpu_launch_ssm_out(int layer, int slot,
    const float* gated_pinned, int gated_dim, int output_dim);

// Block CPU until SSM Out result is in pinned host memory.
// After this returns, slot->h_ssm_out is valid.
int gpu_wait_ssm_out(int slot);
```

### gpu_launch_qkv_gate Implementation

```c
extern "C" int gpu_launch_qkv_gate(int layer, int slot_idx,
    const float* normed_src, int hidden_dim,
    int qkv_dim, int gate_dim)
{
    LayerGPU* lg = &g_layers[layer];
    if (!lg->loaded) return -1;
    PipelineSlot* slot = &g_slots[slot_idx];

    // 1. Copy normed data into pinned buffer (CPU memcpy, fast for 16KB)
    memcpy(slot->h_normed, normed_src, hidden_dim * sizeof(float));

    // 2. Async H2D transfer: pinned -> device
    CHECK_CUDA(cudaMemcpyAsync(slot->d_normed, slot->h_normed,
        hidden_dim * sizeof(float), cudaMemcpyHostToDevice, g_stream_h2d));
    CHECK_CUDA(cudaEventRecord(slot->evt_h2d_normed_done, g_stream_h2d));

    // 3. Compute stream waits for H2D to complete
    CHECK_CUDA(cudaStreamWaitEvent(g_stream_compute, slot->evt_h2d_normed_done, 0));

    // 4. Launch QKV kernel
    q8_matvec_kernel<<<qkv_dim, MATVEC_THREADS, 0, g_stream_compute>>>(
        slot->d_qkv_out, (const unsigned char*)lg->d_qkv,
        slot->d_normed, hidden_dim, qkv_dim);

    // 5. Launch Gate kernel (same stream — serialized after QKV, shares input)
    q8_matvec_kernel<<<gate_dim, MATVEC_THREADS, 0, g_stream_compute>>>(
        slot->d_gate_out, (const unsigned char*)lg->d_gate,
        slot->d_normed, hidden_dim, gate_dim);

    // 6. Record completion of both kernels
    CHECK_CUDA(cudaEventRecord(slot->evt_qkv_gate_done, g_stream_compute));

    // 7. D2H stream waits for compute, then copies results back
    CHECK_CUDA(cudaStreamWaitEvent(g_stream_d2h, slot->evt_qkv_gate_done, 0));
    CHECK_CUDA(cudaMemcpyAsync(slot->h_qkv_out, slot->d_qkv_out,
        qkv_dim * sizeof(float), cudaMemcpyDeviceToHost, g_stream_d2h));
    CHECK_CUDA(cudaMemcpyAsync(slot->h_gate_out, slot->d_gate_out,
        gate_dim * sizeof(float), cudaMemcpyDeviceToHost, g_stream_d2h));
    CHECK_CUDA(cudaEventRecord(slot->evt_d2h_qkv_gate_done, g_stream_d2h));

    return 0;  // Returns immediately — GPU is working asynchronously
}

extern "C" int gpu_wait_qkv_gate(int slot_idx) {
    PipelineSlot* slot = &g_slots[slot_idx];
    CHECK_CUDA(cudaEventSynchronize(slot->evt_d2h_qkv_gate_done));
    return 0;
}
```

### gpu_launch_ssm_out Implementation

```c
extern "C" int gpu_launch_ssm_out(int layer, int slot_idx,
    const float* gated_src, int gated_dim, int output_dim)
{
    LayerGPU* lg = &g_layers[layer];
    if (!lg->loaded) return -1;
    PipelineSlot* slot = &g_slots[slot_idx];

    // 1. Copy gated data into pinned buffer
    memcpy(slot->h_gated_in, gated_src, gated_dim * sizeof(float));

    // 2. Async H2D
    CHECK_CUDA(cudaMemcpyAsync(slot->d_gated_in, slot->h_gated_in,
        gated_dim * sizeof(float), cudaMemcpyHostToDevice, g_stream_h2d));
    CHECK_CUDA(cudaEventRecord(slot->evt_h2d_gated_done, g_stream_h2d));

    // 3. Compute waits for H2D
    CHECK_CUDA(cudaStreamWaitEvent(g_stream_compute, slot->evt_h2d_gated_done, 0));

    // 4. Launch SSM Out kernel
    q8_matvec_kernel<<<output_dim, MATVEC_THREADS, 0, g_stream_compute>>>(
        slot->d_ssm_out, (const unsigned char*)lg->d_ssm_out,
        slot->d_gated_in, gated_dim, output_dim);

    // 5. Record completion
    CHECK_CUDA(cudaEventRecord(slot->evt_ssm_out_done, g_stream_compute));

    // 6. D2H
    CHECK_CUDA(cudaStreamWaitEvent(g_stream_d2h, slot->evt_ssm_out_done, 0));
    CHECK_CUDA(cudaMemcpyAsync(slot->h_ssm_out, slot->d_ssm_out,
        output_dim * sizeof(float), cudaMemcpyDeviceToHost, g_stream_d2h));
    CHECK_CUDA(cudaEventRecord(slot->evt_d2h_ssm_out_done, g_stream_d2h));

    return 0;
}

extern "C" int gpu_wait_ssm_out(int slot_idx) {
    PipelineSlot* slot = &g_slots[slot_idx];
    CHECK_CUDA(cudaEventSynchronize(slot->evt_d2h_ssm_out_done));
    return 0;
}
```

### The New Layer Loop (winmoe_inference.c)

```c
// Replace the inner layer loop with this pipelined version.
// Key change: GPU launches are async, CPU does useful work during GPU execution.

for (int layer = 0; layer < cfg.num_layers; layer++) {
    LayerWeights* lw = &layers[layer];
    int slot_idx = layer % NUM_PIPELINE_SLOTS;  // Alternate between slot 0 and 1
    memcpy(residual, hidden, H * sizeof(float));

    // ── Phase A: Attention Norm ──
    if (lw->attn_norm) {
        rmsnorm(normed, hidden, lw->attn_norm, H);
    } else {
        memcpy(normed, hidden, H * sizeof(float));
    }

    if (lw->is_deltanet && use_gpu && lw->w_qkv) {
        int inner = (model.ssm_inner_size > 0) ? model.ssm_inner_size : 4096;

        // ── Phase B: Launch GPU QKV+Gate (ASYNC — returns immediately) ──
        gpu_launch_qkv_gate(layer, slot_idx, normed, H,
                            inner * 2, inner);

        // ── Phase C: CPU Alpha/Beta (overlaps with GPU QKV+Gate) ──
        // These are tiny projections [4096->32], ~0.05ms each
        float* alpha_raw = dn_states[layer].tmp_alpha;
        float* beta_raw = dn_states[layer].tmp_beta;
        if (lw->w_alpha)
            quant_matvec(alpha_raw, lw->w_alpha, normed, DN_NUM_GATES, H, lw->w_alpha_type);
        if (lw->w_beta)
            quant_matvec(beta_raw, lw->w_beta, normed, DN_NUM_GATES, H, lw->w_beta_type);

        // ── Phase D: SYNC — Wait for QKV+Gate results ──
        // CPU blocks here until GPU results are in pinned host memory.
        // Expected wait: ~1.5ms (GPU 1.6ms - CPU alpha/beta 0.05ms - launch overhead)
        gpu_wait_qkv_gate(slot_idx);

        // Read results from pinned memory
        PipelineSlot* slot = &g_slots[slot_idx];
        float* gpu_qkv = slot->h_qkv_out;   // QKV is in pinned memory — NO COPY needed
        float* gpu_gate = slot->h_gate_out;  // Gate is in pinned memory — NO COPY needed

        // ── Phase E: CPU State Recurrence (1.0ms) ──
        // Split QKV, compute gate params, L2 norm, recurrence
        float* Q = gpu_qkv;
        float* K_ptr = gpu_qkv + DN_INNER;
        float* V_ptr = gpu_qkv + DN_INNER + DN_NUM_KV_GROUPS * DN_HEAD_DIM;

        // Gate parameters
        float gate_decay[DN_NUM_GATES];
        float beta_vals[DN_NUM_GATES];
        for (int hi = 0; hi < DN_NUM_GATES; hi++) {
            float a_val = (lw->ssm_a && lw->ssm_dt_bias) ?
                expf(-expf(lw->ssm_a[hi]) * logf(1.0f + expf(alpha_raw[hi] + lw->ssm_dt_bias[hi])))
                : 0.99f;
            if (a_val > 1.0f) a_val = 1.0f;
            if (a_val < 0.0f) a_val = 0.0f;
            if (_isnan(a_val)) a_val = 0.99f;
            gate_decay[hi] = a_val;
            beta_vals[hi] = 1.0f / (1.0f + expf(-beta_raw[hi]));
        }

        // L2 normalize
        for (int hi = 0; hi < DN_NUM_Q_HEADS; hi++)
            dn_l2norm(Q + hi * DN_HEAD_DIM, DN_HEAD_DIM);
        for (int hi = 0; hi < DN_NUM_KV_GROUPS; hi++)
            dn_l2norm(K_ptr + hi * DN_HEAD_DIM, DN_HEAD_DIM);

        // [... existing AVX2 state recurrence code, operating on gpu_qkv in-place ...]
        // NOTE: gpu_qkv points to PINNED memory (slot->h_qkv_out).
        // The L2 norm modifies it in-place. This is safe because we own this slot
        // and the GPU is done writing to it (we waited on evt_d2h_qkv_gate_done).
        //
        // IMPORTANT: We are modifying pinned memory in-place. This is allowed
        // because pinned memory is both CPU-readable and CPU-writable. The GPU
        // will not touch this buffer again until the NEXT time we call
        // gpu_launch_qkv_gate with this slot (which won't happen until layer+2).

        // ... (full recurrence loop as in current code) ...

        // ── Phase F: RMSNorm + SiLU Gating (0.1ms) ──
        float* head_output = dn_states[layer].tmp_head_out;
        float* normed_out = dn_states[layer].tmp_normed;
        float* gated_out = dn_states[layer].tmp_gated;

        for (int hi = 0; hi < DN_NUM_Q_HEADS; hi++) {
            if (lw->ssm_norm_w)
                dn_rmsnorm_weighted(normed_out + hi * DN_HEAD_DIM,
                    head_output + hi * DN_HEAD_DIM, lw->ssm_norm_w, DN_HEAD_DIM);
            else
                dn_rmsnorm_simple(normed_out + hi * DN_HEAD_DIM,
                    head_output + hi * DN_HEAD_DIM, DN_HEAD_DIM);
        }
        for (int hi = 0; hi < DN_INNER; hi++) {
            float zv = gpu_gate[hi];
            if (zv > 88.0f) zv = 88.0f;
            if (zv < -88.0f) zv = -88.0f;
            gated_out[hi] = normed_out[hi] * (zv / (1.0f + expf(-zv)));
        }

        // ── Phase G: Launch GPU SSM Out (ASYNC) ──
        gpu_launch_ssm_out(layer, slot_idx, gated_out, DN_INNER, H);

        // ── Phase G.1: CPU work while SSM Out runs on GPU (0.4ms window) ──
        // We have ~0.4ms of free CPU time here. Use it for:
        // - Pre-quantize the post-attention activation to Q8_K (for expert FFN)
        //   BUT we don't have the post-attention hidden yet (need SSM Out result).
        // - Instead, we can pre-compute router gate weights (partial).
        //   BUT we need post-attention normed, which needs SSM Out.
        //
        // Actually, there is very little useful CPU work we can do here because
        // everything downstream depends on SSM Out. The 0.4ms is essentially wasted.
        //
        // FUTURE OPTIMIZATION: If we had a second independent computation
        // (e.g., prefetching expert weights based on PREDICTED routing),
        // we could fill this gap. See Section 8 for speculative prefetch.

        // ── Phase H: SYNC — Wait for SSM Out ──
        gpu_wait_ssm_out(slot_idx);

        // Read SSM Out from pinned memory
        memcpy(o_out, slot->h_ssm_out, H * sizeof(float));

    } else if (lw->is_deltanet && lw->w_qkv) {
        // CPU fallback path (unchanged)
        deltanet_forward(o_out, normed, &dn_states[layer], ...);
    } else {
        // Standard attention (see Section 7)
        memset(o_out, 0, H * sizeof(float));
    }

    // ── Phase I: Residual + Post-Norm + Router + Expert FFN ──
    // (Same as current code — entirely CPU, no GPU involvement)
    for (int i = 0; i < H; i++) hidden[i] = residual[i] + o_out[i];
    memcpy(residual, hidden, H * sizeof(float));

    // Post-attention norm
    if (lw->post_attn_norm) {
        rmsnorm(normed, hidden, lw->post_attn_norm, H);
    } else if (lw->ffn_norm) {
        rmsnorm(normed, hidden, lw->ffn_norm, H);
    } else {
        memcpy(normed, hidden, H * sizeof(float));
    }

    // Router topK + Expert FFN (2.9ms total — unchanged)
    // ... (existing code) ...

    // Final residual
    for (int i = 0; i < H; i++) hidden[i] = residual[i] + moe_out[i];
}
```

### Timing Analysis of Pipelined Layer

```
Time:  0ms     0.05ms  1.6ms   1.65ms  2.65ms  2.75ms  3.15ms  3.16ms  6.16ms
       |       |       |       |       |       |       |       |       |
CPU:   [Norm]  [a/b]   [WAIT..........]  [Recur 1.0ms] [Gate] [WAIT]  [Resid+Route+Expert 3.0ms]
GPU:           [=======QKV+Gate========]                [=SSM=]
```

**Per-layer time: ~6.16ms** (vs. current ~6.4ms sequential including overhead)

The savings here are modest: ~0.05ms from alpha/beta overlap, ~0ms elsewhere because
the dependency chain is fundamentally serial within a layer.

---

## 6. Double-Buffering Design

### Why Double-Buffer

Double-buffering ensures that while the GPU writes results for layer L into slot A,
the CPU can still be reading slot B's results from layer L-1.

In our case, however, the pipeline is NOT truly pipelined across layers (see Section 1 analysis).
Each layer must complete before the next starts due to the `hidden_final[L]` dependency.

**Double-buffering is still needed** for a different reason: **preventing pinned memory races**.

When we call `gpu_launch_qkv_gate(layer, slot_idx, ...)`, the function copies `normed`
into `slot->h_normed` and then kicks off an async H2D transfer. If we used slot 0 for
every layer, the CPU's memcpy for layer L+1 could overwrite `h_normed` before the
layer L H2D transfer completes.

With the current design (CPU waits for all GPU results before proceeding), this cannot
happen — each layer fully completes before the next starts. But double-buffering provides
safety margin and enables future optimizations.

### Slot Selection

```c
int slot_idx = layer % NUM_PIPELINE_SLOTS;
```

- Layer 0 -> slot 0
- Layer 1 -> slot 1
- Layer 2 -> slot 0
- Layer 3 -> slot 1
- ...

### Buffer Lifetime Guarantees

For slot S used by layer L:
- `h_normed[S]`: Written by CPU at Phase B, read by GPU H2D transfer, done by Phase D
- `h_qkv_out[S]`: Written by GPU D2H, read by CPU at Phase E, modified in-place (L2 norm)
- `h_gate_out[S]`: Written by GPU D2H, read by CPU at Phase F
- `h_gated_in[S]`: Written by CPU at Phase G, read by GPU H2D transfer, done by Phase H
- `h_ssm_out[S]`: Written by GPU D2H, read by CPU at Phase H

Next use of slot S is layer L+2. By that time, layer L is fully complete.
**No aliasing hazard.**

---

## 7. Standard Attention Layers (Every 4th)

### Layer Pattern in Qwen3.5-397B

```
Layers: 0(DN), 1(DN), 2(DN), 3(SA), 4(DN), 5(DN), 6(DN), 7(SA), ...
```

Standard attention (SA) layers have Q/K/V/O projections + GQA multi-head attention.
Currently they produce zero output (`memset(o_out, 0, ...)`).

### Pipeline Impact

SA layers do NOT use the GPU DeltaNet path. They are currently CPU-only (or stubbed).
When implemented, they will likely need GPU for Q/K/V/O projections.

### Design for SA Layers in the Pipeline

```c
if (!lw->is_deltanet) {
    // Standard attention layer — different GPU work pattern
    //
    // Option A: GPU handles Q/K/V projections, CPU handles attention + O projection
    // Option B: Entire SA layer on CPU (if weights are small enough)
    // Option C: Skip SA layers for now (current behavior)
    //
    // For Phase 3, we implement Option A:
    // 1. GPU: Q projection [4096->4096] (~0.4ms)
    // 2. GPU: K projection [4096->512]  (~0.05ms)
    // 3. GPU: V projection [4096->512]  (~0.05ms)
    // 4. CPU: RoPE + KV cache append + GQA attention (variable, depends on seq len)
    // 5. GPU: O projection [4096->4096] (~0.4ms)
    //
    // This uses the same 3-stream topology. We just need different device buffers
    // for Q/K/V/O (which are smaller than QKV concatenated).

    // For now, SA layers fall through to the synchronous path.
    // They don't interact with the pipeline slots.

    // IMPORTANT: Before entering an SA layer, we must ensure the GPU pipeline
    // is drained (all prior async work complete). This is already guaranteed
    // because we synchronize at the end of each layer.

    memset(o_out, 0, H * sizeof(float));
    // TODO: Implement full GQA on GPU
}
```

### Pipeline Drain at SA Boundaries

When transitioning from a DeltaNet layer to a SA layer (or vice versa), no special
drain is needed because each layer fully synchronizes before returning.

If we later implement cross-layer pipelining, SA layers would act as pipeline "bubbles"
that force a drain and restart.

---

## 8. Expert FFN Overlap with GPU

### The Question

Can expert FFN for layer L overlap with GPU DeltaNet for layer L+1?

### Answer: NO, Due to Data Dependency

As analyzed in Section 1:
```
hidden_final[L] = hidden_after_attn[L] + moe_out[L]
normed[L+1] = RMSNorm(hidden_final[L])
GPU QKV L+1 needs normed[L+1]
```

Layer L+1 cannot begin until `hidden_final[L]` is computed, which requires `moe_out[L]`.

### Alternative: Speculative Expert Prefetch

While we cannot overlap COMPUTE, we CAN overlap **I/O**:

```c
// After router topK for layer L identifies expert_ids[0..3],
// AND if we know layer L+1 is DeltaNet (likely),
// we can start prefetching common/predicted experts for layer L+1
// from SSD into the expert cache.
//
// Heuristic: the same experts are often activated across adjacent layers.
// So after layer L's router selects experts {12, 45, 8, 91}, we can
// speculatively prefetch those same experts for layer L+1.

// This is an SSD I/O optimization, not a GPU pipeline optimization.
// It converts cache misses (~1ms SSD read) into cache hits (0ms).
```

### Expert I/O Prefetch Design

```c
// Add to the expert FFN loop:
// After computing expert FFN for layer L, issue async reads for
// layer L+1's predicted experts.

// Predict: reuse layer L's expert_ids for layer L+1
if (layer + 1 < cfg.num_layers && layers[layer + 1].gate_inp) {
    LayerWeights* lw_next = &layers[layer + 1];
    for (int ek = 0; ek < K; ek++) {
        int predicted_eid = expert_ids[ek];  // Reuse current layer's experts
        int cache_key_next = (layer + 1) * cfg.num_experts + predicted_eid;

        if (!expert_cache[cache_key_next]) {
            // Issue async SSD reads for next layer's predicted experts
            // These will (hopefully) complete by the time we need them.
            // Uses the existing AsyncRead infrastructure with Windows OVERLAPPED I/O.

            // NOTE: This requires pre-allocated async I/O buffers (one per prefetch slot)
            // to avoid malloc in the hot path.
            // See Section 3 for prefetch buffer allocation.
        }
    }
}
```

### Expert Prefetch Buffers

```c
#define MAX_PREFETCH_SLOTS 4  // K=4 experts prefetched

typedef struct {
    AsyncRead ar_gate;
    AsyncRead ar_up;
    AsyncRead ar_down;
    void* slot_buf;       // Pre-allocated expert_total_size buffer
    void* gate_io_buf;    // Aligned I/O buffer
    void* up_io_buf;
    void* down_io_buf;
    int target_cache_key; // Where to store result
    int active;           // 1 if prefetch in flight
} PrefetchSlot;

static PrefetchSlot g_prefetch[MAX_PREFETCH_SLOTS];

// At startup:
for (int p = 0; p < MAX_PREFETCH_SLOTS; p++) {
    g_prefetch[p].slot_buf = malloc(expert_total_size);
    g_prefetch[p].gate_io_buf = _aligned_malloc(expert_buf_size, ALIGN);
    g_prefetch[p].up_io_buf = _aligned_malloc(expert_buf_size, ALIGN);
    g_prefetch[p].down_io_buf = _aligned_malloc(expert_buf_size, ALIGN);
    g_prefetch[p].active = 0;
}

// Before expert FFN for layer L, check if prefetches from layer L-1 are done:
for (int p = 0; p < MAX_PREFETCH_SLOTS; p++) {
    if (g_prefetch[p].active) {
        // Wait for async reads to complete
        async_read_wait(&g_prefetch[p].ar_gate);
        async_read_wait(&g_prefetch[p].ar_up);
        async_read_wait(&g_prefetch[p].ar_down);

        // Store in cache
        int key = g_prefetch[p].target_cache_key;
        if (!expert_cache[key] && cache_stored < max_cached) {
            expert_cache[key] = g_prefetch[p].slot_buf;
            g_prefetch[p].slot_buf = malloc(expert_total_size); // Replace consumed buffer
            cache_stored++;
        }
        g_prefetch[p].active = 0;
    }
}
```

---

## 9. Error Handling and Fault Tolerance

### GPU Slower Than Expected

The async pipeline handles variable GPU timing gracefully because the CPU simply
blocks longer on `cudaEventSynchronize`. There is no deadline — the CPU will wait
as long as needed.

```c
// If GPU takes 3ms instead of 1.6ms for QKV+Gate:
// - CPU does alpha/beta (0.05ms), then blocks for 2.95ms
// - No buffer corruption, no race condition
// - Just slower overall throughput
```

### CUDA Error During Async Execution

CUDA errors during async kernel execution are not detected until the next sync point.
We must check after every `cudaEventSynchronize`:

```c
int gpu_wait_qkv_gate(int slot_idx) {
    PipelineSlot* slot = &g_slots[slot_idx];
    cudaError_t err = cudaEventSynchronize(slot->evt_d2h_qkv_gate_done);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU ERROR in QKV+Gate (slot %d): %s\n",
                slot_idx, cudaGetErrorString(err));
        // Check if this is a sticky error that requires device reset
        cudaError_t last = cudaGetLastError();
        if (last != cudaSuccess) {
            fprintf(stderr, "  Sticky error: %s — falling back to CPU\n",
                    cudaGetErrorString(last));
            // Set flag to disable GPU for remaining layers
            return -1;
        }
        return -1;
    }
    return 0;
}
```

### Fallback to Synchronous Path

```c
// In the layer loop:
int gpu_ok = gpu_launch_qkv_gate(layer, slot_idx, normed, H, inner * 2, inner);
// ... alpha/beta ...
if (gpu_ok == 0) {
    gpu_ok = gpu_wait_qkv_gate(slot_idx);
}

if (gpu_ok != 0) {
    // GPU failed — fall back to CPU-only DeltaNet
    fprintf(stderr, "Layer %d: GPU error, falling back to CPU\n", layer);
    deltanet_forward(o_out, normed, &dn_states[layer], ...);
    use_gpu = 0;  // Disable GPU for all subsequent layers
} else {
    // Continue with GPU results...
}
```

### Pinned Memory Allocation Failure

If `cudaMallocHost` fails (system out of pinned memory), fall back to regular malloc.
Transfers will be slower but still correct:

```c
int pipeline_slots_init(void) {
    for (int s = 0; s < NUM_PIPELINE_SLOTS; s++) {
        PipelineSlot* slot = &g_slots[s];

        // Try pinned, fall back to regular
        if (cudaMallocHost(&slot->h_normed, HIDDEN_DIM * sizeof(float)) != cudaSuccess) {
            fprintf(stderr, "WARN: pinned alloc failed, using regular malloc\n");
            slot->h_normed = (float*)malloc(HIDDEN_DIM * sizeof(float));
            // NOTE: cudaMemcpyAsync with non-pinned memory may silently
            // fall back to synchronous behavior. This is safe but slower.
        }
        // ... same for other buffers ...
    }
}
```

---

## 10. Race Conditions and Deadlock Analysis

### Race Condition 1: Pinned Buffer Overwrite

**Scenario**: CPU writes to `h_normed[slot]` for layer L+2 while GPU is still
reading `h_normed[slot]` for layer L's H2D transfer.

**Protection**: Cannot happen. Layer L fully synchronizes (Phase H) before
returning. Layer L+1 uses the OTHER slot. Layer L+2 uses the same slot as L,
but L is guaranteed complete by then.

**Formal argument**: Slot S is used by layers {S, S+2, S+4, ...}. Layer N completes
all GPU work before control returns to the CPU to start layer N+1. Therefore,
when layer N+2 starts and writes to slot S, layer N's use of slot S is finished.

### Race Condition 2: In-Place Modification of Pinned QKV Buffer

**Scenario**: The CPU modifies `h_qkv_out[slot]` in-place (L2 normalization of Q and K)
after the D2H transfer completes. Meanwhile, the GPU might be reading from `d_qkv_out[slot]`.

**Protection**: The GPU only reads `d_qkv_out[slot]` during the QKV kernel, which
finishes before the D2H transfer (the D2H waits on `evt_qkv_gate_done`). By the time
the CPU receives the data, the GPU is done with `d_qkv_out[slot]`. And `h_qkv_out[slot]`
is only used for D2H transfer (GPU writes to it, then CPU reads/modifies).
No conflict because the D2H event ensures temporal ordering.

### Race Condition 3: d_normed Shared Between QKV and Gate Kernels

**Scenario**: Both QKV and Gate kernels read from `d_normed[slot]`. Are they safe?

**Protection**: Both kernels are launched on the SAME stream (`g_stream_compute`), so
they execute sequentially. The QKV kernel finishes before the Gate kernel starts.
Both only READ `d_normed`, so even concurrent reads would be safe, but sequential
execution is guaranteed.

### Race Condition 4: Concurrent Use of d_input_fp32/d_output_fp32 (Legacy)

**Scenario**: The old synchronous API (`gpu_deltanet_projections`) uses global
`d_input_fp32` and `d_output_fp32`. If the async API is mixed with the sync API,
both would write to the same global buffers.

**Protection**: The new async API uses per-slot device buffers (`slot->d_normed`,
`slot->d_qkv_out`, etc.) and NEVER touches `d_input_fp32`/`d_output_fp32`.
The old sync API can be kept for the fallback path. The two paths are mutually
exclusive (if `use_gpu_async` is set, we use the async path; otherwise, sync).

**Action**: Add a global flag `g_async_mode` and guard all access to the global
device buffers:

```c
static int g_async_mode = 0;

// In sync path:
if (g_async_mode) { fprintf(stderr, "BUG: sync API called in async mode\n"); return -1; }
```

### Race Condition 5: cudaEventSynchronize vs. Next cudaEventRecord

**Scenario**: We record event E on stream S, then synchronize on E from the CPU.
Then we record E again on stream S for the next layer. Is there a race where the
second `cudaEventRecord` overwrites E before `cudaEventSynchronize` completes?

**Protection**: `cudaEventSynchronize` blocks the CPU thread until the event fires.
After it returns, the CPU thread has exclusive control. The subsequent `cudaEventRecord`
is a CPU-side call that modifies the event's internal state. This is safe because:
1. The old event has already fired (synchronize returned).
2. CUDA events support re-recording (overwriting previous state).
3. The GPU is not referencing the old event anymore (it was consumed by `cudaStreamWaitEvent`
   which has already unblocked).

No race.

### Deadlock Analysis

**Potential deadlock 1**: CPU waits for GPU event that was never recorded.

**Prevention**: Every `gpu_launch_*` function records all events unconditionally.
If the kernel launch fails (returns -1), the wait function will detect the error.
We add a timeout mechanism:

```c
// Enhanced wait with timeout (prevents infinite hang):
int gpu_wait_qkv_gate_timeout(int slot_idx, int timeout_ms) {
    PipelineSlot* slot = &g_slots[slot_idx];
    // cudaEventSynchronize has no timeout. We must poll instead:
    LARGE_INTEGER freq, start, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);

    while (1) {
        cudaError_t err = cudaEventQuery(slot->evt_d2h_qkv_gate_done);
        if (err == cudaSuccess) return 0;
        if (err != cudaErrorNotReady) return -1;  // Real error

        QueryPerformanceCounter(&now);
        double elapsed_ms = (double)(now.QuadPart - start.QuadPart) / freq.QuadPart * 1000.0;
        if (elapsed_ms > timeout_ms) {
            fprintf(stderr, "GPU TIMEOUT: QKV+Gate took >%dms\n", timeout_ms);
            return -2;  // Timeout — fall back to CPU
        }
        // Brief yield to avoid busy-spin burning CPU:
        _mm_pause();  // ~30 cycles on x86
    }
}
```

**Potential deadlock 2**: Circular wait between streams.

**Prevention**: Our stream dependency graph is acyclic:
```
stream_h2d -> (event) -> stream_compute -> (event) -> stream_d2h
```
No cycle. `stream_d2h` never waits on `stream_h2d`. `stream_compute` never waits
on `stream_d2h`. Only forward dependencies.

**Potential deadlock 3**: CPU waits for GPU event while holding a lock that the
GPU callback needs.

**Prevention**: We use no locks and no GPU callbacks. The CPU thread is single-threaded.
No mutexes, no condition variables, no potential for deadlock.

---

## 11. VRAM Budget

### Current Usage: 6.1 GB

```
Q8_0 weights for 45 DeltaNet layers:
  QKV:  45 × (4096 × 8192 / 32) × 34 = 45 × 34,816 × 34 ≈ 50.7 MB per layer → 2,282 MB
  Gate: 45 × (4096 × 4096 / 32) × 34 = 45 × 17,408 × 34 ≈ 25.3 MB per layer → 1,141 MB
  SSM:  45 × (4096 × 4096 / 32) × 34 ≈ 25.3 MB per layer → 1,141 MB

  Subtotal: ~4,564 MB (Q8_0, not FP16)

  Plus: d_input_fp32, d_output_fp32, d_input_fp16, d_output_fp16
  ~256 KB

  Total: ~4.5 GB actual (the 6.1 GB in the prompt may include driver overhead or FP16 weights)
```

### New VRAM for Pipeline Buffers

```
Per slot (2 slots):
  d_normed:   4096 × 4 = 16 KB
  d_qkv_out:  8192 × 4 = 32 KB
  d_gate_out: 4096 × 4 = 16 KB
  d_gated_in: 4096 × 4 = 16 KB
  d_ssm_out:  4096 × 4 = 16 KB
  Total per slot: 96 KB

2 slots: 192 KB
```

**Impact**: 192 KB / 8192 MB = 0.002%. Negligible.

### Can We Remove Old Buffers?

Once the async pipeline is fully operational, the old `d_input_fp32` and `d_output_fp32`
global device buffers (max 16384 * 4 * 2 = 128 KB) can be freed. But keeping them
for the fallback path costs almost nothing.

---

## 12. Implementation Phases

### Phase 3a: Async GPU Infrastructure (2-3 days)

**Changes to `gpu_offload.cu`:**
1. Add stream creation/destruction to `gpu_init()`/`gpu_shutdown()`
2. Add `PipelineSlot` struct and `pipeline_slots_init()`/`pipeline_slots_free()`
3. Implement `gpu_launch_qkv_gate()`, `gpu_wait_qkv_gate()`
4. Implement `gpu_launch_ssm_out()`, `gpu_wait_ssm_out()`
5. Add timeout-based polling variant

**Changes to `gpu_offload.h`:**
1. Add new async function declarations
2. Add `pipeline_init()`/`pipeline_free()` declarations
3. Add `NUM_PIPELINE_SLOTS` constant

**Testing:**
- Unit test: launch QKV+Gate async, do CPU work, wait, verify results match sync path
- Regression test: ensure sync path still works (fallback)
- Stress test: run 100 tokens, verify no memory corruption

### Phase 3b: Integrate Async Pipeline into Inference Loop (1-2 days)

**Changes to `winmoe_inference.c`:**
1. Add `pipeline_init()` call after `gpu_init()`
2. Replace synchronous `gpu_deltanet_projections()` + `gpu_ssm_out_projection()`
   with async launch/wait pattern
3. Remove per-layer `malloc(inner * 2 * sizeof(float))` for gpu_qkv — use pinned slot buffer instead
4. Remove per-layer `malloc(inner * sizeof(float))` for gpu_gate — use pinned slot buffer instead
5. Add error handling with fallback to sync path

**Key Refactor:**
```c
// OLD (lines 684-688):
float* gpu_qkv = (float*)malloc(inner * 2 * sizeof(float));
float* gpu_gate = (float*)malloc(inner * sizeof(float));
gpu_deltanet_projections(layer, normed, H, gpu_qkv, inner * 2, gpu_gate, inner);

// NEW:
int slot_idx = layer % NUM_PIPELINE_SLOTS;
gpu_launch_qkv_gate(layer, slot_idx, normed, H, inner * 2, inner);
// ... CPU alpha/beta work ...
gpu_wait_qkv_gate(slot_idx);
float* gpu_qkv = g_slots[slot_idx].h_qkv_out;  // Zero-copy from pinned memory
float* gpu_gate = g_slots[slot_idx].h_gate_out;
```

This eliminates 2 malloc+free calls per DeltaNet layer (90 allocations per token).

### Phase 3c: Expert Prefetch (1-2 days)

**Changes to `winmoe_inference.c`:**
1. Allocate prefetch slots at startup
2. After router topK for layer L, issue async SSD reads for predicted layer L+1 experts
3. Before expert FFN for layer L, harvest any completed prefetches from layer L-1
4. Store prefetched data in expert cache

### Phase 3d: Measurement and Tuning (1 day)

1. Add per-phase timing (GPU wait time, CPU work time, overlap efficiency)
2. Compare async vs sync path performance
3. Tune: should Gate kernel be on a SEPARATE compute stream from QKV?
   (Test: launch QKV and Gate concurrently on two streams. On RTX 3070,
   the two kernels are both memory-bound, so concurrent execution likely
   does NOT help. But worth measuring.)
4. Measure prefetch hit rate and SSD I/O overlap benefit

---

## 13. Expected Performance Model

### Per DeltaNet Layer — Async Pipeline

```
Phase A: RMSNorm             0.05ms  (CPU)
Phase B: Launch QKV+Gate     0.01ms  (CPU, async launch overhead)
Phase C: Alpha/Beta          0.05ms  (CPU, overlaps with GPU)
Phase D: Wait QKV+Gate       1.54ms  (CPU blocked, GPU working 1.6ms - 0.06ms overlap)
Phase E: State recurrence    1.00ms  (CPU)
Phase F: RMSNorm+SiLU gate   0.10ms  (CPU)
Phase G: Launch SSM Out      0.01ms  (CPU, async launch overhead)
Phase G.1: Wasted GPU time   0.00ms  (nothing to overlap)
Phase H: Wait SSM Out        0.39ms  (CPU blocked, GPU working 0.4ms - 0.01ms launch)
Phase I: Residual+Norm       0.06ms  (CPU)
Phase J: Router topK         0.40ms  (CPU)
Phase K: Expert FFN (cached) 2.50ms  (CPU)
Phase M: Residual add        0.01ms  (CPU)
                              ─────
Total per DeltaNet layer:     6.12ms
```

### Comparison with Current

```
Current:  6.4ms per DeltaNet layer (with sync overhead)
Async:    6.12ms per DeltaNet layer
Savings:  0.28ms per layer × 45 layers = 12.6ms per token
```

**Disappointing. The async pipeline alone yields only ~3% improvement.**

### Where the Real Wins Come From

The async pipeline's primary benefit is NOT overlap (the dependency chain is too tight).
The real wins are:

1. **Eliminating per-layer malloc/free (Phase 3b)**: Removes 90 malloc+90 free calls per token.
   Estimated savings: 0.5-1.0ms per token (malloc is expensive on Windows, especially
   for pinned-adjacent patterns).

2. **Expert prefetch (Phase 3c)**: Eliminates SSD cache misses after warmup.
   Estimated savings: When cache is cold, each miss costs ~1ms.
   With prefetch, 60-80% of misses become hits.
   During steady-state (cache warm), savings are ~0ms.

3. **Async GPU launch amortization**: The CUDA driver batches commands more efficiently
   when using explicit streams vs. implicit synchronize-after-every-call.
   Current code calls `cudaMemcpy` (synchronous) 4 times per layer = 180 sync calls.
   New code calls `cudaEventSynchronize` 2 times per layer = 90 sync calls.
   Estimated savings: 0.01-0.02ms per eliminated sync × 90 = 0.9-1.8ms per token.

4. **Preparing for multi-token batching**: The async infrastructure is a prerequisite
   for processing multiple tokens in parallel (batch size > 1), where GPU work for
   token T+1 can truly overlap with CPU work for token T.

### Revised Total Savings Estimate

```
Async GPU (overlap):              12.6ms
Eliminate malloc/free:             1.0ms
Reduce sync point count:           1.5ms
Expert prefetch (cold start):     variable
                                  ─────
Total estimated:                  ~15ms per token

384ms → ~369ms per token = ~2.71 tok/s
```

### Path to 4.8 tok/s (210ms target)

The async pipeline alone cannot reach 4.8 tok/s. Additional optimizations needed:

1. **Expert FFN on GPU** (biggest win): Move K=4 expert matvecs to GPU.
   Currently 2.5ms on CPU per layer. GPU Q4_K matvec could do it in ~0.5ms.
   Savings: 2.0ms × 60 layers = 120ms per token.
   This makes the pipeline: 3.6ms/layer × 60 = 216ms -> ~4.6 tok/s.
   Requires: Expert weights in VRAM (won't fit all, but hot experts could be cached).

2. **Speculative execution**: Run multiple layer pipelines concurrently by
   predicting `hidden_final[L]` (e.g., assume moe_out is small compared to hidden).
   Risky but potentially high reward.

3. **Kernel fusion**: Fuse QKV+Gate into single kernel launch (eliminate launch overhead).
   Fuse RMSNorm + alpha/beta with QKV input prep.

4. **VNNI/AVX-512 for state recurrence**: The 1.0ms recurrence uses AVX2 FMA.
   AVX-512 could potentially halve this to 0.5ms.

---

## Appendix A: Complete Modified gpu_offload.h

```c
#pragma once
/*
 * GPU Offload for DeltaNet Projections — Phase 3 Async Pipeline
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize GPU: streams, cuBLAS, pipeline slots */
int gpu_init(void);

/* Initialize pipeline double-buffer slots (call after gpu_init) */
int gpu_pipeline_init(void);

/* Upload one DeltaNet layer's weights (Q8_0 on GPU) */
int gpu_upload_deltanet_weights(int layer,
    const void* qkv_q8, int qkv_rows, int qkv_cols,
    const void* gate_q8, int gate_rows, int gate_cols,
    const void* ssm_out_q8, int ssm_rows, int ssm_cols);

/* === SYNCHRONOUS API (legacy, fallback) === */

int gpu_deltanet_projections(int layer,
    const float* normed, int hidden_dim,
    float* qkv_out, int qkv_dim,
    float* gate_out, int gate_dim);

int gpu_ssm_out_projection(int layer,
    const float* gated, int gated_dim,
    float* output, int output_dim);

/* === ASYNCHRONOUS API (Phase 3 pipeline) === */

/* Launch QKV+Gate projections. Returns immediately.
 * Results will be in pinned host memory accessible via gpu_slot_qkv/gate(). */
int gpu_launch_qkv_gate(int layer, int slot,
    const float* normed, int hidden_dim,
    int qkv_dim, int gate_dim);

/* Block until QKV+Gate results are ready. Returns 0 on success, -1 on error. */
int gpu_wait_qkv_gate(int slot);

/* Get pointers to QKV/Gate results in pinned host memory (valid after wait). */
float* gpu_slot_qkv(int slot);
float* gpu_slot_gate(int slot);

/* Launch SSM Out projection. Returns immediately. */
int gpu_launch_ssm_out(int layer, int slot,
    const float* gated, int gated_dim, int output_dim);

/* Block until SSM Out result is ready. */
int gpu_wait_ssm_out(int slot);

/* Get pointer to SSM Out result in pinned host memory. */
float* gpu_slot_ssm_out(int slot);

/* Cleanup */
void gpu_pipeline_free(void);
void gpu_shutdown(void);

/* Stats */
int gpu_is_initialized(void);
float gpu_vram_used_mb(void);

#define GPU_NUM_PIPELINE_SLOTS 2

#ifdef __cplusplus
}
#endif
```

## Appendix B: Complete Modified gpu_offload.cu Additions

```cuda
/* ═══════════════════════════════════════════════════════════════════
 * Phase 3: Async Pipeline Infrastructure
 * ═══════════════════════════════════════════════════════════════════ */

#define HIDDEN_DIM_MAX 4096
#define QKV_DIM_MAX    8192
#define GATE_DIM_MAX   4096
#define SSM_OUT_DIM_MAX 4096

/* CUDA streams */
static cudaStream_t g_stream_compute = NULL;
static cudaStream_t g_stream_h2d = NULL;
static cudaStream_t g_stream_d2h = NULL;

/* Pipeline slot */
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

    /* Synchronization events */
    cudaEvent_t evt_h2d_normed_done;
    cudaEvent_t evt_qkv_gate_done;
    cudaEvent_t evt_d2h_qkv_gate_done;
    cudaEvent_t evt_h2d_gated_done;
    cudaEvent_t evt_ssm_out_done;
    cudaEvent_t evt_d2h_ssm_out_done;

    int initialized;
} PipelineSlot;

static PipelineSlot g_slots[GPU_NUM_PIPELINE_SLOTS];
static int g_pipeline_initialized = 0;

extern "C" int gpu_pipeline_init(void) {
    if (g_pipeline_initialized) return 0;
    if (!g_initialized) return -1;

    /* Create streams */
    CHECK_CUDA(cudaStreamCreateWithFlags(&g_stream_compute, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&g_stream_h2d, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&g_stream_d2h, cudaStreamNonBlocking));

    /* Allocate slots */
    for (int s = 0; s < GPU_NUM_PIPELINE_SLOTS; s++) {
        PipelineSlot* slot = &g_slots[s];

        /* Pinned host */
        CHECK_CUDA(cudaMallocHost(&slot->h_normed,   HIDDEN_DIM_MAX * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&slot->h_qkv_out,  QKV_DIM_MAX * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&slot->h_gate_out,  GATE_DIM_MAX * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&slot->h_gated_in, GATE_DIM_MAX * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&slot->h_ssm_out,  SSM_OUT_DIM_MAX * sizeof(float)));

        /* Device */
        CHECK_CUDA(cudaMalloc(&slot->d_normed,   HIDDEN_DIM_MAX * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&slot->d_qkv_out,  QKV_DIM_MAX * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&slot->d_gate_out,  GATE_DIM_MAX * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&slot->d_gated_in, GATE_DIM_MAX * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&slot->d_ssm_out,  SSM_OUT_DIM_MAX * sizeof(float)));

        /* Events — disable timing for lower overhead */
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_h2d_normed_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_qkv_gate_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_d2h_qkv_gate_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_h2d_gated_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_ssm_out_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&slot->evt_d2h_ssm_out_done, cudaEventDisableTiming));

        slot->initialized = 1;
    }

    g_pipeline_initialized = 1;
    fprintf(stderr, "Pipeline initialized: %d slots, 3 streams, VRAM +%.0f KB\n",
            GPU_NUM_PIPELINE_SLOTS,
            GPU_NUM_PIPELINE_SLOTS * 5.0f * HIDDEN_DIM_MAX * 4 / 1024);
    return 0;
}

extern "C" int gpu_launch_qkv_gate(int layer, int slot_idx,
    const float* normed_src, int hidden_dim,
    int qkv_dim, int gate_dim)
{
    LayerGPU* lg = &g_layers[layer];
    if (!lg->loaded) return -1;
    if (slot_idx < 0 || slot_idx >= GPU_NUM_PIPELINE_SLOTS) return -1;
    PipelineSlot* slot = &g_slots[slot_idx];
    if (!slot->initialized) return -1;

    /* Copy normed into pinned buffer */
    memcpy(slot->h_normed, normed_src, hidden_dim * sizeof(float));

    /* H2D: normed -> device */
    CHECK_CUDA(cudaMemcpyAsync(slot->d_normed, slot->h_normed,
        hidden_dim * sizeof(float), cudaMemcpyHostToDevice, g_stream_h2d));
    CHECK_CUDA(cudaEventRecord(slot->evt_h2d_normed_done, g_stream_h2d));

    /* Compute waits for H2D */
    CHECK_CUDA(cudaStreamWaitEvent(g_stream_compute, slot->evt_h2d_normed_done, 0));

    /* QKV kernel */
    q8_matvec_kernel<<<qkv_dim, MATVEC_THREADS, 0, g_stream_compute>>>(
        slot->d_qkv_out, (const unsigned char*)lg->d_qkv,
        slot->d_normed, hidden_dim, qkv_dim);

    /* Gate kernel */
    q8_matvec_kernel<<<gate_dim, MATVEC_THREADS, 0, g_stream_compute>>>(
        slot->d_gate_out, (const unsigned char*)lg->d_gate,
        slot->d_normed, hidden_dim, gate_dim);

    /* Record compute done */
    CHECK_CUDA(cudaEventRecord(slot->evt_qkv_gate_done, g_stream_compute));

    /* D2H waits for compute, then transfers results */
    CHECK_CUDA(cudaStreamWaitEvent(g_stream_d2h, slot->evt_qkv_gate_done, 0));
    CHECK_CUDA(cudaMemcpyAsync(slot->h_qkv_out, slot->d_qkv_out,
        qkv_dim * sizeof(float), cudaMemcpyDeviceToHost, g_stream_d2h));
    CHECK_CUDA(cudaMemcpyAsync(slot->h_gate_out, slot->d_gate_out,
        gate_dim * sizeof(float), cudaMemcpyDeviceToHost, g_stream_d2h));
    CHECK_CUDA(cudaEventRecord(slot->evt_d2h_qkv_gate_done, g_stream_d2h));

    return 0;
}

extern "C" int gpu_wait_qkv_gate(int slot_idx) {
    PipelineSlot* slot = &g_slots[slot_idx];
    cudaError_t err = cudaEventSynchronize(slot->evt_d2h_qkv_gate_done);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU error in gpu_wait_qkv_gate(slot %d): %s\n",
                slot_idx, cudaGetErrorString(err));
        cudaGetLastError(); /* Clear sticky error */
        return -1;
    }
    return 0;
}

extern "C" float* gpu_slot_qkv(int slot_idx) {
    return g_slots[slot_idx].h_qkv_out;
}

extern "C" float* gpu_slot_gate(int slot_idx) {
    return g_slots[slot_idx].h_gate_out;
}

extern "C" int gpu_launch_ssm_out(int layer, int slot_idx,
    const float* gated_src, int gated_dim, int output_dim)
{
    LayerGPU* lg = &g_layers[layer];
    if (!lg->loaded) return -1;
    if (slot_idx < 0 || slot_idx >= GPU_NUM_PIPELINE_SLOTS) return -1;
    PipelineSlot* slot = &g_slots[slot_idx];
    if (!slot->initialized) return -1;

    /* Copy gated into pinned buffer */
    memcpy(slot->h_gated_in, gated_src, gated_dim * sizeof(float));

    /* H2D */
    CHECK_CUDA(cudaMemcpyAsync(slot->d_gated_in, slot->h_gated_in,
        gated_dim * sizeof(float), cudaMemcpyHostToDevice, g_stream_h2d));
    CHECK_CUDA(cudaEventRecord(slot->evt_h2d_gated_done, g_stream_h2d));

    /* Compute waits for H2D */
    CHECK_CUDA(cudaStreamWaitEvent(g_stream_compute, slot->evt_h2d_gated_done, 0));

    /* SSM Out kernel */
    q8_matvec_kernel<<<output_dim, MATVEC_THREADS, 0, g_stream_compute>>>(
        slot->d_ssm_out, (const unsigned char*)lg->d_ssm_out,
        slot->d_gated_in, gated_dim, output_dim);

    /* Record */
    CHECK_CUDA(cudaEventRecord(slot->evt_ssm_out_done, g_stream_compute));

    /* D2H */
    CHECK_CUDA(cudaStreamWaitEvent(g_stream_d2h, slot->evt_ssm_out_done, 0));
    CHECK_CUDA(cudaMemcpyAsync(slot->h_ssm_out, slot->d_ssm_out,
        output_dim * sizeof(float), cudaMemcpyDeviceToHost, g_stream_d2h));
    CHECK_CUDA(cudaEventRecord(slot->evt_d2h_ssm_out_done, g_stream_d2h));

    return 0;
}

extern "C" int gpu_wait_ssm_out(int slot_idx) {
    PipelineSlot* slot = &g_slots[slot_idx];
    cudaError_t err = cudaEventSynchronize(slot->evt_d2h_ssm_out_done);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU error in gpu_wait_ssm_out(slot %d): %s\n",
                slot_idx, cudaGetErrorString(err));
        cudaGetLastError();
        return -1;
    }
    return 0;
}

extern "C" float* gpu_slot_ssm_out(int slot_idx) {
    return g_slots[slot_idx].h_ssm_out;
}

extern "C" void gpu_pipeline_free(void) {
    if (!g_pipeline_initialized) return;

    for (int s = 0; s < GPU_NUM_PIPELINE_SLOTS; s++) {
        PipelineSlot* slot = &g_slots[s];
        if (!slot->initialized) continue;

        cudaFreeHost(slot->h_normed);
        cudaFreeHost(slot->h_qkv_out);
        cudaFreeHost(slot->h_gate_out);
        cudaFreeHost(slot->h_gated_in);
        cudaFreeHost(slot->h_ssm_out);

        cudaFree(slot->d_normed);
        cudaFree(slot->d_qkv_out);
        cudaFree(slot->d_gate_out);
        cudaFree(slot->d_gated_in);
        cudaFree(slot->d_ssm_out);

        cudaEventDestroy(slot->evt_h2d_normed_done);
        cudaEventDestroy(slot->evt_qkv_gate_done);
        cudaEventDestroy(slot->evt_d2h_qkv_gate_done);
        cudaEventDestroy(slot->evt_h2d_gated_done);
        cudaEventDestroy(slot->evt_ssm_out_done);
        cudaEventDestroy(slot->evt_d2h_ssm_out_done);

        slot->initialized = 0;
    }

    if (g_stream_compute) { cudaStreamDestroy(g_stream_compute); g_stream_compute = NULL; }
    if (g_stream_h2d)     { cudaStreamDestroy(g_stream_h2d);     g_stream_h2d = NULL; }
    if (g_stream_d2h)     { cudaStreamDestroy(g_stream_d2h);     g_stream_d2h = NULL; }

    g_pipeline_initialized = 0;
}
```

## Appendix C: Timing Instrumentation

```c
/* Add to winmoe_inference.c for Phase 3 profiling */

typedef struct {
    double gpu_qkv_gate_launch_ms;   /* Time to launch async QKV+Gate */
    double cpu_alpha_beta_ms;         /* CPU alpha/beta overlap time */
    double gpu_qkv_gate_wait_ms;     /* Time CPU blocked waiting for QKV+Gate */
    double cpu_recurrence_ms;         /* State recurrence time */
    double cpu_gating_ms;             /* RMSNorm + SiLU gating time */
    double gpu_ssm_out_launch_ms;    /* Time to launch async SSM Out */
    double gpu_ssm_out_wait_ms;      /* Time CPU blocked waiting for SSM Out */
    double cpu_expert_ms;            /* Router + Expert FFN time */
    double cpu_overhead_ms;          /* Residual adds, norms, etc */
} LayerProfile;

/* Accumulate across layers: */
LayerProfile layer_prof;
memset(&layer_prof, 0, sizeof(layer_prof));

/* After QKV+Gate launch: */
QPC(&t0);
gpu_launch_qkv_gate(...);
QPC(&t1);
layer_prof.gpu_qkv_gate_launch_ms += ELAPSED(t0, t1);

/* After alpha/beta: */
QPC(&t0);
quant_matvec(alpha_raw, ...);
quant_matvec(beta_raw, ...);
QPC(&t1);
layer_prof.cpu_alpha_beta_ms += ELAPSED(t0, t1);

/* Before and after wait: */
QPC(&t0);
gpu_wait_qkv_gate(slot_idx);
QPC(&t1);
layer_prof.gpu_qkv_gate_wait_ms += ELAPSED(t0, t1);

/* At end of token: */
fprintf(stderr, "  Pipeline: launch=%.2fms alpha=%.2fms wait_qkv=%.2fms "
        "recur=%.2fms gate=%.2fms wait_ssm=%.2fms expert=%.2fms\n",
        layer_prof.gpu_qkv_gate_launch_ms,
        layer_prof.cpu_alpha_beta_ms,
        layer_prof.gpu_qkv_gate_wait_ms,
        layer_prof.cpu_recurrence_ms,
        layer_prof.cpu_gating_ms,
        layer_prof.gpu_ssm_out_wait_ms,
        layer_prof.cpu_expert_ms);

/* KEY METRIC: overlap_efficiency = cpu_alpha_beta_ms / gpu_qkv_gate_wait_ms
 * If close to 0: CPU work is negligible compared to GPU wait (no overlap benefit)
 * If close to 1: CPU work perfectly fills the GPU wait (maximum overlap)
 * Expected: ~0.03 (0.05ms / 1.6ms) — very low, confirming the honest analysis
 */
```

## Appendix D: Honest Assessment and Next Steps

### What This Phase Actually Achieves

The dependency analysis shows that GPU-CPU pipeline overlap for the DeltaNet architecture
provides **modest throughput gains** (~3-5%) from:

1. Eliminating synchronous `cudaMemcpy` (using async transfers instead)
2. Removing per-layer heap allocations (using pre-allocated pinned pools)
3. Reducing driver-side synchronization overhead (fewer sync points)
4. Overlapping the tiny alpha/beta CPU projections with GPU QKV+Gate

The fundamental bottleneck is the **serial dependency chain**: GPU must finish before
CPU recurrence can start, and CPU must finish before GPU SSM Out can start. There is
no way to pipeline across layers because `hidden_final[L]` depends on `moe_out[L]`.

### What Would Actually Reach 4.8 tok/s

The path to 4.8 tok/s requires attacking the **2.5ms Expert FFN** which is 47% of
per-token time and currently CPU-only. Options:

1. **GPU Expert FFN**: Keep hot experts in VRAM (~200 MB for top-50 most common).
   Route to GPU if expert is in VRAM, CPU otherwise. This can cut expert time by 2-4x
   for cached experts.

2. **AVX-512 Expert FFN**: The i7-11800H supports AVX-512 + VNNI. The current Q4_K
   dequant uses AVX2. AVX-512 could potentially provide 1.5-2x speedup for expert matvec.

3. **Reduce K from 4 to 2**: Halves expert FFN time from 2.5ms to 1.25ms per layer.
   Quality impact must be measured.

4. **State recurrence on GPU**: Move the 1.0ms AVX2 recurrence to a custom CUDA kernel.
   This eliminates the GPU->CPU->GPU round-trip and could reduce DeltaNet time to
   GPU-only: QKV(0.8ms) + Gate(0.8ms) + Recurrence(0.3ms) + SSM_Out(0.4ms) = 2.3ms/layer.
   Saves: (current 3.6ms - 2.3ms) = 1.3ms × 45 layers = 58.5ms per token.

### Priority Order

```
Priority  Optimization                  Expected Gain   Effort
1         AVX-512 Expert FFN            ~30-40ms        Medium (kernel port)
2         Phase 3 async pipeline        ~15ms           Medium (this plan)
3         GPU state recurrence          ~58ms           Hard (custom CUDA kernel for 32 heads × 128×128)
4         GPU Expert FFN (hot cache)    ~60-80ms        Hard (VRAM management, routing logic)
5         Reduce K to 2                 ~75ms           Easy (config change, quality test needed)
```

The async pipeline (this plan) is Priority 2 because it is a prerequisite for items 3 and 4,
and its infrastructure (streams, pinned memory, events) will be reused by all subsequent
GPU optimizations.
