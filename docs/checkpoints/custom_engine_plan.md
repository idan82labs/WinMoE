# Custom Expert Service Engine — Architecture Plan

## Bottleneck

At 1.7 tok/s (588 ms/token), the dominant cost is **expert-byte service**: reading expert
weights from SSD, staging through RAM, and transferring to GPU. llama.cpp uses mmap with
OS-managed page faults — this means:
- No control over read granularity (4KB page faults vs 3.3MB expert blocks)
- No async prefetch (faults are synchronous, blocking)
- No pinned memory (pageable H2D at 6-8 GB/s, not 18-19 GB/s)
- No pipeline overlap (can't read layer N+1 while computing layer N)

## Engine thesis

> Explicit control of the SSD->RAM->GPU pipeline beats OS-managed mmap by at least 2x
> on expert-byte service time.

By replacing mmap with explicit offset-based reads into pinned staging buffers with async
GPU transfer and tiered caching, we can hide most of the SSD latency behind GPU compute.

## Model parameters (IQ2_XXS quantization)

- Total model on disk: ~107 GB (4 GGUF shards)
- Expert weights: ~100 GB (~93% of total)
- Total experts: 512 per layer × 60 layers = 30,720
- Per expert block: ~3.3 MB (IQ2_XXS)
- Active experts per token per layer: K=3 (current best) → 10 MB/layer
- Shared weights (attention, embeddings, norms, routing): ~7 GB (always resident)

## Timing budget (current, measured)

- Total per token: 588 ms (1.7 tok/s at K=3)
- Expert I/O estimate: K×expert_size×layers / SSD_bandwidth = 3×3.3MB×60 / 2.1GB/s ≈ **283 ms**
- With mmap overhead (page faults, non-sequential): likely **350-450 ms**
- Compute + attention + routing: **~140-240 ms**

## Staged milestones

| Stage | Target | Metric |
|-------|--------|--------|
| 0 | Measure current llama.cpp service path | Per-layer timing breakdown |
| 1 | 2x lower expert-byte service time | ms/expert-set on replayed traces |
| 2 | 2.5 tok/s at K=3 or 2.0 tok/s at K=5 | End-to-end tok/s |
| 3 | 3.0 tok/s at K=3 or 2.5 tok/s at K=5 | End-to-end tok/s |

## Slab format specification

```
Per-layer slab file: layer_XX.slab (XX = 00..59)
  Header: 4 bytes magic "SLAB"
  Header: 4 bytes version (1)
  Header: 4 bytes num_experts (512)
  Header: 4 bytes expert_size_bytes
  Header: 4 bytes alignment (4096)
  Index: num_experts × 8 bytes (offset, size) pairs
  Padding to 4096 boundary
  Expert 0: [expert_size_bytes, padded to 4096]
  Expert 1: [expert_size_bytes, padded to 4096]
  ...
  Expert 511: [expert_size_bytes, padded to 4096]

Shared weights file: shared.bin
  Attention weights, embeddings, norms, routing matrices
  Loaded once at startup, resident in GPU VRAM + RAM

Index file: expert_index.json
  { "layer_00": { "path": "layer_00.slab", "num_experts": 512,
    "expert_size": 3342336, "aligned_size": 3342336, "experts": [
      {"id": 0, "offset": 4096, "size": 3342336}, ... ] }, ... }
```

All expert blocks 4KB-aligned for FILE_FLAG_NO_BUFFERING compatibility.
Sequential layout within each layer for maximum SSD sequential read performance.

## Benchmark metrics per workstream

| Workstream | Primary metric | Target |
|------------|---------------|--------|
| engine-replay | Simulated tok/s on replayed traces | 3.0+ tok/s simulated |
| engine-repacker | Sequential read MB/s on repacked vs GGUF | >2.5 GB/s |
| engine-io | MB/s on expert-sized reads (3.3 MB) at queue depth 16-32 | >2.0 GB/s |
| engine-staging | End-to-end SSD→pinned RAM→GPU latency per expert set | <5 ms |
| engine-gpu | Expert dequant+compute per layer | <2 ms |
| engine-cache | Hit rate at VRAM(1000)/RAM(5000) boundaries | >90% |

## Current baseline to beat

- 1.7 tok/s = 588 ms/token
- ~350-450 ms is expert I/O (mmap, uncontrolled)
- ~140-240 ms is compute + attention
- Target: reduce expert I/O from ~400ms to <200ms → 2.5+ tok/s
