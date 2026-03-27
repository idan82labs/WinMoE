# Professor Questions V3 — Custom Engine Architecture

We are building a custom MoE inference engine from scratch. No mmap. No llama.cpp
loader. Full control over the SSD→RAM→GPU pipeline. The professor's answers will
lock the architecture before we write code.

## Hardware constraints (fixed)
- NVMe: Samsung 980, 2.1 GB/s sequential, 42 us/request overhead
- RAM: 40 GB DDR4 (30 GB usable after OS)
- GPU: RTX 3070 Laptop, 8 GB VRAM, PCIe Gen4 x16 (19 GB/s pinned H2D)
- CPU: 16 logical cores, AVX512

## Model parameters (fixed)
- Qwen3.5-397B-A17B, IQ2_XXS quantization
- 60 layers, 512 experts/layer, K=3 active per token
- Expert block: 3.5 MB each
- Total expert data: ~100 GB (30,720 expert blocks)
- Shared weights (attention, embeddings, routing): ~7 GB (always resident)
- Current best: 1.7 tok/s (llama.cpp mmap, CPU-only AVX512)
- Compute ceiling: 2.34 tok/s (professor-derived, T_c = 7.12 ms/layer)

## Measured I/O performance
- mmap page faults: 586 MB/s effective (4.9 us/page overhead)
- Explicit unbuffered reads: 2,200 MB/s (3.7x faster)
- Pinned H2D transfer: 19 GB/s
- Cache hit rate (warm, 500 tokens): ~67% VRAM+RAM combined

---

## Q1: Optimal Memory Architecture — No mmap

With full control over memory, we choose how to manage 100 GB of expert data
in 30 GB usable RAM + 8 GB VRAM.

**Design**: Three-tier explicit cache:
- **VRAM tier** (C_v blocks): pinned in GPU memory, zero-latency access
- **RAM tier** (C_r blocks): in cudaHostAlloc pinned memory, 19 GB/s to GPU
- **SSD tier** (remaining): read on-demand via explicit unbuffered I/O

No page faults. No OS page cache. No virtual memory tricks. We own every byte.

**Questions**:

1. What is the **optimal (C_v, C_r) allocation** given:
   - Total VRAM budget for experts: ~6 GB (after shared weights)
   - Total RAM budget for experts: ~22 GB (after shared weights + OS)
   - Expert block size: 3.5 MB
   - C_v ≈ 1,714 blocks, C_r ≈ 6,285 blocks → covers 8,000/30,720 = 26%
   - With Zipf s=0.46 routing: what hit rate does this give?

2. Should the RAM tier use **static LFU** (pin the globally most frequent experts)
   or **per-sequence adaptive** (evict/promote based on current routing)?

3. The professor's earlier answer said "adaptive-heavy early, static-heavy late."
   In a custom engine where we control eviction explicitly, what is the **exact
   switchover point** (in tokens) where static begins to dominate adaptive?

4. With explicit memory management, is there value in **over-provisioning** the
   RAM cache with swap-like behavior — keep more than 22 GB of experts "soft-pinned"
   and let the OS page out cold ones? Or does this recreate the mmap problem?

---

## Q2: Pipeline Scheduling — Can We Achieve max(T_c, T_io)?

The theoretical optimum is:
```
t_layer = max(T_compute, T_io_miss)
```
instead of the causal:
```
t_layer = T_compute + T_io_miss
```

In a custom engine, we control the execution order. Proposed pipeline:

```
Layer N:    [GPU compute expert FFN] ──────────────────→
Layer N+1:  [CPU: route → identify experts] → [SSD read → pinned RAM] ──→
Layer N+2:                                              [SSD prefetch] ──→
```

**Questions**:

1. **Is the pipeline achievable on separate PCIe lanes?** The NVMe controller and
   GPU are on different PCIe endpoints. Can SSD reads truly run in parallel with
   GPU compute without contention? Or does the CPU memory controller become the
   bottleneck when both SSD DMA and GPU DMA target system RAM simultaneously?

2. **What is the minimum pipeline depth** needed to fully hide I/O behind compute?
   With T_c = 7.12 ms and T_io = 5.0 ms (K=3 × 3.5MB / 2.1 GB/s), a depth of 1
   should suffice (T_io < T_c). But with cache misses varying per layer, what
   depth handles the **worst case** (all K=3 experts are cold)?

3. **Routing is known only after attention completes for layer N.** This means
   expert IDs for layer N+1 are available only after ~2 ms into layer N's compute.
   With T_io = 5 ms, we have 7.12 - 2 = 5.12 ms of lead time. Is this enough?
   If not, what **prediction recall** do we need to start reads earlier?

4. The professor's crossover formula gave B_crossover = 1.39 GB/s. **With pipeline
   overlap, does the crossover change?** In the pipelined case:
   ```
   t_token = L × max(T_c, T_io_effective)
   ```
   What is T_io_effective after caching (67% hit rate)?
   ```
   T_io_effective = 0.33 × K × S_e / B = 0.33 × 3 × 3.5MB / 2.1GB/s = 1.65 ms
   ```
   Since T_io_effective (1.65 ms) << T_c (7.12 ms), are we **already fully
   compute-bound** in the pipelined regime even at 2.1 GB/s?

---

## Q3: Can We Break the 2.34 tok/s Ceiling?

The ceiling TPS_max = 1/(L × T_c) = 2.34 assumes T_c = 7.12 ms/layer measured
on the old binary. The source-built AVX512 binary runs at 1.8 tok/s CPU-only
(vs 1.22 on old), suggesting T_c dropped ~32%.

**Questions**:

1. **What is the hardware roofline for T_c on RTX 3070 + AVX512 CPU?**
   The expert FFN at K=3, IQ2_XXS is: dequant 3 × 3.5MB + SwiGLU + combine.
   What is the theoretical minimum compute time per layer given:
   - CPU: AVX512, ~200 GFLOPS FP32
   - GPU: RTX 3070, 20.3 TFLOPS FP32 (but only 8 GB VRAM)
   - Expert FFN at IQ2_XXS: mostly integer dequant, not FP matmul

2. **If T_c can be reduced to 5 ms** (via better CUDA kernels for dequant):
   - New ceiling: 1/(60 × 0.005) = 3.33 tok/s
   - Is this realistic? What would it take?

3. **Batch-2 inference**: Can we process 2 tokens in parallel to amortize
   attention compute? MoE routing is per-token, but attention K/V cache is
   shared. If batch-2 halves attention cost per token:
   - New effective T_c = T_expert + T_attention/2
   - What fraction of T_c is attention vs expert compute?
   - Is batch-2 viable with 8 GB VRAM + 30 GB RAM?

4. **Speculative decoding with MoE**: Standard speculative decoding uses a
   small draft model to propose N tokens, then verifies with the large model.
   For MoE, the draft model could predict BOTH the next token AND which experts
   it will need, enabling prefetch. Is this information-theoretically viable
   given the 24.4-bit label entropy the professor calculated?

---

## Q4: NVMe Queue Isolation on Windows

The M/G/1-PS contention model assumed a single service queue. NVMe hardware
has 64+ submission queues. Windows StorNVMe driver maps I/O to queues.

**Questions**:

1. If we open two file handles (one for attention weights, one for expert slabs)
   with different I/O patterns (sequential vs random), **does Windows assign them
   to different NVMe submission queues?**

2. If we use overlapped I/O (IOCP) with queue depth 32 on one handle while
   synchronous reads happen on another, **do they contend or parallelize?**

3. **Is there a Windows API to explicitly control NVMe queue assignment?**
   DirectStorage may offer this. Is it worth integrating DirectStorage for
   the expert read path specifically?

4. If queue isolation IS possible: the pipeline model becomes:
   ```
   Queue 1 (priority): expert reads for current layer
   Queue 2 (background): prefetch for layer N+2
   ```
   **Can we assign priority levels to different queues on consumer NVMe?**

---

## Q5: Optimal Slab Layout for Sequential NVMe Reads

We're building a slab repacker. Need to decide the on-disk layout.

**Options**:

**A. Per-layer files** (60 files, ~1.7 GB each):
```
layer_00.slab: [expert_0][expert_1]...[expert_511]  (512 × 3.5 MB = 1.75 GB)
```
Pro: one sequential read per layer, clean alignment
Con: 60 file handles, OS overhead

**B. Single file, layer-ordered** (1 file, 100 GB):
```
[layer_0_exp_0..511][layer_1_exp_0..511]...[layer_59_exp_0..511]
```
Pro: single file handle, sequential access pattern matches layer order
Con: huge file, harder to update

**C. Single file, frequency-ordered** (1 file, hot experts first):
```
[hottest 1000 experts across all layers][next 1000]...[coldest]
```
Pro: hot data is at low offsets → better NVMe prefetch
Con: breaks layer-sequential access pattern

**Questions**:

1. At 3.5 MB block size, **does NVMe internal prefetch/read-ahead matter?**
   Our blocks are already larger than the NVMe page size (4 KB) and the
   controller's internal read-ahead window. Does layout matter at this granularity?

2. For the custom engine with tiered cache, **which layout minimizes cold-start
   time** (loading the initial hot set into RAM cache)?

3. **Alignment**: should expert blocks be aligned to 4 KB (NVMe page), 64 KB
   (common read-ahead chunk), or 1 MB (large page)? What's the optimal alignment
   for FILE_FLAG_NO_BUFFERING reads at our block size?

---

## Q6: The 40 GB RAM Constraint — Expert Cache Sizing

With 30 GB usable for expert cache (after shared weights and OS):
- C_r = 30 GB / 3.5 MB ≈ 8,571 blocks out of 30,720 total = 27.9%

The professor's miss curve: M(C) ≈ 1 - (C^0.54 - 1)/(512^0.54 - 1) per layer.

**Questions**:

1. With per-layer cache of C_r/60 ≈ 143 blocks per layer (out of 512):
   - Static LFU hit rate: h ≈ (143^0.54 - 1)/(512^0.54 - 1) ≈ ?
   - Miss rate per layer: M ≈ ?
   - Expected SSD reads per token: 60 × K × M = ?

2. **Should the cache be per-layer or global?** Per-layer wastes capacity on
   layers with uniform routing (low skew). Global allows hot layers to use
   more cache. What is the expected gain from global vs per-layer allocation?

3. With 67% measured warm hit rate and explicit I/O for misses:
   ```
   T_io_per_token = 60 × 0.33 × 3 × 3.5MB / 2100 MB/s = 99 ms
   ```
   Combined with T_compute = 427 ms:
   ```
   t_token_causal = 427 + 99 = 526 ms → 1.90 tok/s
   t_token_pipeline = max(427, 99) = 427 ms → 2.34 tok/s
   ```
   **Is pipeline overlap the only path to 2.34, or can cache improvements
   alone get us there without pipeline complexity?**

---

## Priority Order

1. **Q2** (pipeline scheduling) — determines if we can reach 2.34 or stay at 1.9
2. **Q3** (breaking 2.34) — determines the ultimate ceiling
3. **Q1** (memory architecture) — determines cache design
4. **Q4** (NVMe queues) — determines if pipeline is physically achievable
5. **Q6** (cache sizing) — determines RAM allocation
6. **Q5** (slab layout) — determines disk format

## What We Need From the Professor

- **Lock the architecture** before we write 10,000 lines of engine code
- **Identify showstoppers** — anything that makes the custom engine NOT worth building
- **Quantify the ceiling** — what is the absolute best tok/s achievable on this hardware?
- **Pipeline feasibility** — can SSD + GPU truly run in parallel on discrete PCIe?
