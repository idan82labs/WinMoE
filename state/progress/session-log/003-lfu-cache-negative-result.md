# Session 003 — LFU cache eviction: negative result; pivot to compute-side levers

**Date:** 2026-05-17
**Outcome:** WINMOE_CACHE_POLICY=lfu added; two variants tested vs round-robin baseline. Both LOSE on tok/s despite higher hit rate. Phase 4.2 (cache eviction) deprioritized — bottleneck is FFN compute.

---

## Measurements (P1, 234 positions = 34 prompt + 200 gen, identical workload)

| Policy | Cache hit rate | tok/s | Per-token (attn / expert / total) |
|---|---|---|---|
| Round-robin (baseline) | **52.3%** | **0.33** | 650 / 1700 / 3060 ms |
| LFU pure | 19.9% | 0.20 | 1100 / 3300 / 5070 ms |
| LFU-DA (seeded inserts) | 58.5% | 0.26 | 920 / 2170 / 3870 ms |

LFU-DA does improve hit rate (+6pp) but tok/s *regresses* by 21%.

## Why LFU loses

1. **OS page-cache interaction.** Round-robin rotates evictions, so a freshly-evicted expert is often still in the Windows file-cache; a "miss" costs ~0.3ms instead of ~3ms. LFU permanently evicts the truly-cold long tail, pushing those misses to the disk-cold path where each miss costs 10× more.
2. **Compute dominates IO.** Per-token breakdown shows expert ~1700ms of which IO wait is only a small fraction. Eviction policy only affects IO; can't reduce CPU/GPU FFN compute time.

## Decisions made

- Keep `WINMOE_CACHE_POLICY=lfu` available as env-gated diagnostic but **revert default to round-robin**.
- Stop investing in cache-eviction sophistication (2Q, ARC, LIRS) for THIS hardware profile. The math doesn't work — even with 90% hit rate, the 1700ms of FFN compute is unchanged.
- Re-aim Phase 4 effort at compute-side levers:
  1. **More experts on GPU** (`MAX_GPU_EXPERTS` 300 → 500; ~1.5GB more VRAM, ~10× compute speedup per migrated expert)
  2. **Phase 4.4 prefetch (SCOPE-A')** — overlaps IO with compute; gains only if IO is on the critical path, which is only ~25% of expert time currently
  3. **Phase 4.3 DN-MoE-skip** — skip MoE every-other layer if PPL holds; ~2× speedup if safe; risk it costs quality
  4. **Phase 4.1 IQ2 cold quant** — smaller cold-tier means bigger effective cache footprint; modest (~2-5%)

## Honest path to 1.5 tok/s
- Current: 0.33 tok/s
- More GPU experts (×1.2): 0.40
- Prefetch (×1.15): 0.46
- IQ2 cold (×1.1): 0.51
- DN-MoE-skip if safe (×2.0): **1.0 tok/s**

That stack lands at ~1.0 tok/s — still short of 1.5. Hitting 1.5 likely requires either DN-MoE-skip being safer than 2× (e.g., skip 2 of 3), or a fundamental kernel speedup (e.g., direct CUDA Q4_K_M expert matmul rather than dequant-then-matmul). Or accepting 1.0 tok/s ceiling on this hardware.

## What's next
- Bump MAX_GPU_EXPERTS from 300 → 500, measure VRAM headroom + tok/s delta.
- If positive, look at Phase 4.4 prefetch design.
- DN-MoE-skip is highest reward / highest risk — needs careful PPL measurement before committing.
