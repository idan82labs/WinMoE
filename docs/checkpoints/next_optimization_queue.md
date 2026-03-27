# Optimization Queue — Post 397B First Success

Baseline: **1.00 tok/s at K=5, CPU-only, IQ2_XXS, Samsung 980 NVMe**

All trials below are ordered by expected impact / effort ratio.
Each trial follows the autoresearch loop pattern: one change, measure, keep or reject.

---

## Tier 1 — Immediate (no code changes, <5 min each)

### Trial 4: GPU offload for attention layers
- Change: `--n-gpu-layers 5` then `10` then `15`
- Theory: Qwen3.5 has 60 layers. Attention (non-MoE) is compute-bound. Offloading attention to RTX 3070 frees CPU for expert weight processing.
- Expected: 10-30% speedup on compute-bound portion
- Risk: VRAM OOM if too many layers. 8 GB VRAM limits how many layers fit.
- Spec reference: Section 6 (VRAM hybrid split)

### Trial 5: Thread count optimization
- Change: `--threads 4`, `--threads 8`, `--threads 12`
- Theory: llama.cpp defaults to all logical cores. MoE inference may benefit from fewer threads (less contention on memory bus during expert loading).
- Expected: 5-15% change (direction unknown)
- Risk: none

### Trial 6: K=6 measurement
- Change: `patch_k_experts.py --k 6`
- Theory: fills the gap between K=5 (1.0 tok/s) and K=7 (0.82 tok/s). K=6 likely ~0.9 tok/s with better quality than K=5.
- Expected: ~0.9 tok/s, useful data point for K_l optimization
- Risk: none

### Trial 7: Context reduction
- Change: `--ctx 256` (from 512)
- Theory: smaller context = less KV cache RAM = more room for OS to cache expert weights
- Expected: slight speedup from reduced memory pressure
- Risk: limits usable context length

---

## Tier 2 — Short-term (small patches, hours)

### Trial 8: Layerwise K_l schedule
- Change: modify llama.cpp or GGUF to use different K per layer
- Theory (Spec Section 8.3): input/output layers are more quality-sensitive. Use K=8 on layers 0-4 and 55-59, K=3 on layers 20-40. Average K stays ~5 but quality improves.
- Formula: `min sum c_l(K_l) s.t. sum epsilon_l(K_l) <= epsilon_max`
- Expected: same tok/s as uniform K=5 but measurably better output quality
- Risk: requires per-layer K support (may need llama.cpp source modification)
- Spec reference: Section 8, Loop 4

### Trial 9: Warm expert pre-loading
- Change: before generation, run a dummy forward pass to populate OS page cache with hot experts
- Theory (Spec Section 4.5): static LFU hotset can be pre-warmed. First token is slow (14s at K=10) partly because experts aren't cached yet.
- Expected: 30-50% reduction in TTFT
- Risk: adds startup latency

### Trial 10: Carry-forward expert predictor
- Change: modify inference to predict next-layer experts = current-layer experts
- Theory (Spec Section 9): OLMoE showed 37.5% consecutive overlap. If 3/8 experts are predictable, prefetching saves ~37% of SSD wait time.
- Expected: 15-25% decode speedup
- Risk: requires llama.cpp modification or custom wrapper

---

## Tier 3 — Medium-term (build from source, days)

### Trial 11: CUDA build of llama.cpp
- Change: build llama-cpp-python with `-DGGML_CUDA=on`
- Prerequisites: CUDA toolkit, CMake, Visual Studio Build Tools
- Theory: enables true GPU offload (attention + expert compute on GPU) and async SSD→GPU pipeline
- Expected: 2-3x overall speedup from GPU acceleration
- Spec reference: Sections 6, 9 (VRAM split, overlap model)

### Trial 12: Better quantization (Q4_K_S)
- Change: download Q4_K_S (228 GB) to replace IQ2_XXS (107 GB)
- Theory: Q4 has dramatically better quality than IQ2. Perplexity difference is typically 2-5 points.
- Expected: much better output quality, similar or slightly slower speed (larger expert blocks)
- Trade-off: needs 228 GB disk space (currently have ~140 GB free on D:)

### Trial 13: Pinned memory expert staging
- Change: use cudaHostAlloc for expert weight buffers
- Theory (Spec WS0): pinned memory achieves 19 GB/s H2D vs 7 GB/s pageable
- Expected: if CUDA build works, 2.7x faster expert → GPU transfer
- Risk: requires CUDA build (Trial 11)

---

## Tier 4 — Long-term (engineering projects, weeks)

### Trial 14: Custom expert streaming engine
- Theory (full V2 spec): replace llama.cpp's mmap-based expert loading with explicit `ReadFile(FILE_FLAG_NO_BUFFERING)` + pinned staging + cudaMemcpyAsync pipeline
- Expected: full pipeline overlap, approaching the `max(T_c, t_ssd)` model instead of `T_c + t_ssd`
- Spec reference: entire V2 spec

### Trial 15: Expert slab repacking
- Theory (Spec WS0): repack experts as contiguous slabs ordered by access frequency
- Expected: maximize sequential SSD bandwidth (2.9 GB/s at 64MB vs 90 MB/s at 4KB)

### Trial 16: Predictive prefetch with ML router
- Theory (Spec Section 9, WS6): train a lightweight predictor that forecasts next-layer expert needs
- FlashMoE paper achieved 51% hit rate improvement over LRU
- Expected: significant miss reduction on top of static caching

---

## Progress Tracking

| Trial | Status | Result | Decision |
|-------|--------|--------|----------|
| 1 (K=10 cold) | done | 0.44 tok/s | baseline |
| 2 (K=7 cold) | done | 0.82 tok/s | keep |
| 3 (K=5 cold) | done | 1.00 tok/s | keep |
| 3b (K=5 warm) | done | 1.24 tok/s | **new warm baseline** |
| 4 (GPU offload) | done | 1.27 tok/s | **reject** — no effect (CPU-only build), speed = warm cache |
| 4b (K=4 warm) | done | 1.31 tok/s | keep (data point) |
| 5 (Thread tuning) | done | 0.98 tok/s @12t | **reject** — no improvement over default |
| 6 (K=6 warm) | done | 1.15 tok/s | keep (data point) |
| 6b (K=8 warm) | done | 0.85 tok/s | keep (data point) |
| 7 (ctx=256) | done | 1.25 tok/s | **reject** — no improvement over ctx=512 |
| 8 (Layerwise K_l) | queued | — | — |
| 9 (Warm pre-load) | done (implicit) | +24% from cache warming | **proven** — page cache matters |
| 10 (Carry-forward) | queued | — | needs llama.cpp mod |
| 11 (CUDA build) | **HIGH PRIORITY** | — | biggest remaining lever |
| 12 (Q4 quant) | queued | — | quality improvement |
| 13 (Pinned memory) | queued | — | needs CUDA build |
| 14 (Custom engine) | queued | — | — |
| 15 (Slab repacking) | queued | — | — |
| 16 (ML predictor) | queued | — | — |

## Warm-cache K sweep (complete)

| K | tok/s (warm) | ms/token | TTFT (ms) |
|---|-------------|----------|-----------|
| 4 | 1.31 | 763 | 6,664 |
| 5 | 1.24 | 806 | 7,519 |
| 6 | 1.15 | 870 | 8,127 |
| 7 | (est ~1.00) | ~1000 | ~9,000 |
| 8 | 0.85 | 1,181 | 9,603 |
| 10 | (est ~0.65) | ~1500 | ~11,000 |

## Key findings from Tier 1 trials

1. **Page cache warming gives 24% free speedup** (1.00 -> 1.24 tok/s). This is the "warm start" lever from the spec and it's already working via OS page cache.
2. **GPU offload has NO effect** with CPU-only llama.cpp build. CUDA build (Trial 11) is the single most important remaining optimization.
3. **Thread count and context size don't matter** at current operating point.
4. **K scaling is remarkably linear**: each unit of K costs ~80-100 ms/token in warm cache regime.
