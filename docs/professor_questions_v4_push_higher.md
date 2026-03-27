# Professor Questions V4 — Push Beyond 3 tok/s

The streaming scheduler simulation shows results ABOVE your predictions.
We need to validate these numbers and find the path to push higher.

## New measured/simulated data

| K | Cache | Streamed tok/s | Pure compute floor |
|---|-------|---------------|-------------------|
| 3 | 67% | 3.04 | 3.25 (307.8 ms) |
| 5 | 67% | 4.10 | 3.26 (307.2 ms) |
| 8 | 67% | 5.04 | 3.26 |
| 10 | 67% | 5.48 | 3.26 |
| 3 | 0% (all SSD) | 2.29 | 3.25 |

The source-built AVX512 binary gives T_c ≈ 5.13 ms/layer (not 7.12 ms).
Per-expert compute: 1.71 ms at K=3, 1.024 ms at K=5, 0.64 ms at K=8.

---

## Q1: Are the K=5+ Numbers Real or a Simulation Artifact?

At K=5, the simulator shows 4.10 tok/s. At K=10, 5.48 tok/s. These are
HIGHER than the pure compute floor for K=3 (3.25 tok/s).

This happens because the simulator computes `total_time = sum over layers of
schedule_layer()`, and with more experts per layer, each expert's compute
is shorter (5.12ms / K), creating better overlap ratios.

**But is this physically achievable?**

The compute floor at K=5 is also 307.2 ms (60 × 5 × 1.024) — same total
compute regardless of K. So why does K=5 show 4.10 tok/s (244 ms/token)
which is BELOW the compute floor?

Wait — 244 ms < 307 ms. That means the simulator is showing token times
below the compute floor. **Is this a bug in the simulation, or does the
streaming pipeline genuinely allow sub-compute-floor token times?**

The pipeline model: within each layer, experts compute sequentially but I/O
overlaps with compute. If all K experts are cached (VRAM/RAM), the per-layer
time is K × t_compute_per_expert. But 5 × 1.024 = 5.12 ms regardless.

**Please verify**: is the streaming model double-counting something, or is
there a real mechanism by which K=5 tokens complete faster than K=3 tokens
despite doing more total compute?

## Q2: What Is the Actual Hardware Compute Floor?

The simulator uses T_c = 5.12 ms/layer as total MoE FFN compute. But this
was measured from llama.cpp's Stage 0, which includes:
- Expert weight dequantization (IQ2_XXS → FP32)
- Gate/up/down matrix multiplications
- SwiGLU activation
- Expert combine + residual

**In a custom engine**, some of these could be faster:
1. CUDA dequant kernels instead of CPU AVX512
2. Fused dequant+matmul kernels
3. GPU-side expert combine (avoid CPU round-trip)

**What is the theoretical minimum T_c for this model on RTX 3070?**

Expert FFN at IQ2_XXS K=3:
- 3 experts × (gate + up + down) = 9 matrix-vector multiplications
- Each matrix: ~1 MB (IQ2_XXS compressed)
- Dequant to FP16/FP32 then matmul with hidden state (dim=4096)

Is this compute-bound or memory-bandwidth-bound on RTX 3070 (192 GB/s)?

## Q3: Optimal K for the Custom Engine

The streaming model shows a surprising result: **higher K is faster in tok/s**
because the per-expert pipeline overlap improves. But higher K also means:
- More SSD reads per layer (partially cached)
- Higher quality output
- More total compute (same per-layer, but more experts contribute)

**Is there an optimal K that maximizes tok/s × quality?**

If we define a quality proxy as retained gate mass p(K), and speed as the
simulated tok/s, the product p(K) × tok/s gives:
- K=3: 0.63 × 3.04 = 1.92
- K=5: 0.78 × 4.10 = 3.20
- K=8: 0.93 × 5.04 = 4.69
- K=10: 1.00 × 5.48 = 5.48

**K=10 dominates on quality×speed.** This is the opposite of what we
assumed — we thought lower K was better for speed. Is this correct, or
does the simulation miss a real-world bottleneck that appears at higher K?

## Q4: Memory Bandwidth Contention at Higher K

At K=10 with 67% cache (33% SSD miss), per layer:
- 3.3 SSD reads × 3.5 MB = 11.55 MB from SSD
- 6.7 RAM reads × 3.5 MB = 23.45 MB from RAM → GPU (via PCIe)
- 10 expert computes × 0.512 ms = 5.12 ms compute

The PCIe transfer for 23.45 MB: 23.45 / 19000 = 1.23 ms
The SSD read for 11.55 MB: 11.55 / 2100 = 5.5 ms

But SSD and PCIe transfers happen simultaneously with compute in the
streaming pipeline. **Do SSD DMA and PCIe H2D DMA contend for the
CPU memory controller?** Both write to system RAM.

If they contend: the effective bandwidth drops and the simulation is
too optimistic at high K.

If they don't contend: K=10 at 5.48 tok/s is achievable.

## Q5: Integration Path — What's the Minimum Viable Engine?

We have 4 validated components:
1. Slab file (107 GB, 2031 MB/s reads)
2. I/O reader (explicit unbuffered, measured)
3. Tiered cache (48-67% hit rates)
4. Streaming scheduler (3.04 tok/s at K=3 simulated)

To wire these into actual inference, we need to replace llama.cpp's
weight loading with our slab reader + cache + scheduler. The compute
kernels (matmul, dequant, attention) still come from GGML/llama.cpp.

**What is the minimum integration that proves the speedup?**

Options:
A. Modify llama.cpp to use our slab reader for expert tensors only
B. Build a standalone engine using GGML as a library
C. Use llama-cpp-python with a custom weight provider callback

**Which has the lowest risk and fastest time-to-measurement?**

## Q6: Does the Q4_K_M Quantization Change the Picture?

All our measurements use IQ2_XXS (2-bit). At Q4_K_M (4-bit):
- Expert block size: ~6.3 MB (vs 3.5 MB)
- SSD read per expert: 3.0 ms (vs 1.67 ms)
- More I/O per expert → streaming ratio changes
- Better quality → K reduction less necessary

**Does Q4 shift the optimal operating point?**

At K=3, Q4:
- Cold load: 3.0 + 0.33 = 3.33 ms/expert
- Compute: probably similar or slightly higher (more data to dequant)
- Ratio: load > compute → streaming less effective

**Is there a quantization level where the streaming pipeline breaks
down** because load time dominates compute time?

---

## Priority

1. **Q1** — validate simulation numbers (are they real or a bug?)
2. **Q3** — optimal K (counterintuitive result needs verification)
3. **Q4** — memory bandwidth contention (key feasibility question)
4. **Q2** — compute floor (determines ceiling)
5. **Q5** — integration path (next engineering step)
6. **Q6** — Q4 quantization impact (future upgrade)
