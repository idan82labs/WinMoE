# Professor Answers V3 — Custom Engine Architecture (LOCKED)

## The Critical Correction

The "prefetch layer N+1 while computing layer N" model is WRONG for this architecture.
Expert IDs are produced INSIDE the current layer's MLP path (after attention + norm + router).
Layer-ahead prefetch requires prediction, not exact knowledge.

**The correct primitive**: within-layer K-expert streaming in residency order.
1. Run the gate (router)
2. Sort 3 selected experts by residency: VRAM first, RAM second, SSD last
3. Start compute on the first ready expert
4. Stream remaining experts behind it

## VRAM Budget Correction

Shared weights (~7 GB) + 8 GB VRAM = NO room for 1,714 expert blocks.
Realistic VRAM expert cache: **150-300 blocks only**.
RAM is the primary cache, VRAM is staging + tiny hotset.

## Pipeline: Expert-Streaming Within Layer (NOT Layer Lookahead)

Physical overlap IS feasible (overlapped I/O + CUDA streams + pinned memory).
GPUDirect Storage NOT available on Windows — path is SSD→RAM→GPU.

Key ratio:
- Cold expert load: 1.85 ms (1.67 SSD + 0.18 H2D)
- Per-expert compute: 1.71 ms (5.12ms / 3)
- These are nearly equal — streaming works!

**Projected tok/s with expert-streaming engine:**
- All 3 cold: ~1.80 tok/s
- 67% RAM hit, no VRAM: ~2.26 tok/s
- + modest VRAM fraction: ~2.29 tok/s
- Optimized: **2.25-2.34 tok/s reachable**

## Ceiling: 2.34 is NOT a hard hardware limit

It's the current compute limit at T_c = 7.12 ms/layer. To break it:
1. Lower T_c (kernel optimization)
2. Accepted multi-token decoding
3. Higher aggregate throughput (multiple sequences)

**Practical ceiling**: 2.25-2.30 safe target, 2.34 best-case, beyond requires kernel wins.
Do NOT freeze architecture assuming 3.33 tok/s (T_c=5ms) — that's aspirational.

## Memory Architecture (LOCKED)

**Key asymmetry**: RAM hit saves 1.67 ms (SSD), VRAM hit saves only 0.184 ms (H2D).
RAM block is worth **9.1x** more than VRAM block.

**Policy**:
- Maximize C_r first
- Keep C_v small and surgical (150-300 blocks)
- VRAM = staging + tiny hotset, NOT main cache

**RAM cache policy (LOCKED)**:
- 75-85% static LFU
- 15-25% per-sequence adaptive
- Adaptive matters most in first 40-60 tokens, then static dominates

**Do NOT over-provision and let OS page out.** Hard user-space budget only.

## Cache Sizing

C_r = 8,571 blocks (30 GB / 3.5 MB), 143/layer = 49% static LFU hit.
Measured 67% warm hit = strong temporal/sequence locality beyond stationary Zipf.

**Can cache alone reach 2.34 without pipeline? NO.**
No-overlap: 427 + 99 + 33 = 559 ms = 1.79 tok/s. Pipeline is necessary.

## NVMe Queue Isolation: DON'T COUNT ON IT

Two file handles do NOT guarantee two hardware NVMe queues.
IOCP is software, not hardware queue mapping.
Mixing IOCP depth-32 + sync reads still contends.
DirectStorage exists but adds D3D12/CUDA interop complexity.

**Architecture answer**: One async demand I/O engine + one best-effort prefetch class.
Prefetch only uses spare queue depth, never equal priority with demand.

## Slab Layout (LOCKED)

**Single file, layer-ordered, hotness-minor within layer.**
- One slab file, fixed-size slots
- Layer-major order
- Within each layer: experts sorted by hotness (most frequent first)
- 64 KiB slot alignment (3.5 MiB = 56 × 64 KiB — perfect fit)

Beats per-layer files (no handle churn) and global frequency order (runtime demand is layer-local).

Cold start: hotset manifest in RAM, bulk preload reads sorted by offset.

## Architecture to Lock (Professor's Recommendation)

1. **Exact, no-prediction engine** — no speculative expert prefetch
2. **Unbuffered overlapped Win32 I/O** — aligned host buffers
3. **CUDA streams** — H2D + compute overlap
4. **RAM = primary expert cache** — 75-85% static LFU + 15-25% adaptive
5. **VRAM = small staging/hot tier** — 150-300 blocks max
6. **No OS-paged cache overflow** — hard user-space budget
7. **Within-layer K-expert streaming** — residency-ordered, not layer-ahead
8. **One demand I/O engine + one best-effort prefetch** — prefetch uses slack only
9. **Single slab file** — 64 KiB slots, layer-major, hotness-minor
10. **In-RAM offset index** — for O(1) expert block lookup

## Planning Numbers

| Scenario | tok/s |
|----------|-------|
| Current (llama.cpp mmap) | 1.7 |
| Base custom engine (all cold) | 1.80 |
| With 67% RAM cache | 2.26 |
| + VRAM hot tier | 2.29 |
| Best-case exact engine | **2.34** |
| Stretch (kernel wins) | high 2s |
| T_c=5ms (aspirational) | 3.33 |
