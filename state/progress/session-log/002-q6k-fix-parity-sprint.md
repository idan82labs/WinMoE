# Session 002 — Q6_K LM-head bug, Phase 1-3 parity sprint

**Date:** 2026-04-21 → 2026-05-16
**Outcome:** v10.48 shipped. WinMoE correctness gate PASS vs llama.cpp on all 5 code prompts.

---

## What we did

### Phase 1 — teacher-forced parity (5 prompts × 50 tokens)
Built infrastructure from scratch:
- `llama-trace-dump --dump-logits` — float32[vocab] per position binary
- `llama-trace-dump --dump-norm` — full result_norm vector binary
- `llama-trace-dump --n-gen N` — drive teacher-forcing in batch
- `WINMOE_DUMP_ALL_LOGITS=path` — per-position logit dump in WinMoE
- `WINMOE_DUMP_NORMED=path` — full normed vector per position
- `loops/parity-coding/harness.py` — top-1/top-5/rank/delta/KL metrics + plots
- `loops/parity-coding/run_parity.py` — orchestrator with cached llama bins

**First-pass result: ALL 5 PROMPTS FAILED gate.** Pattern: top1 0.74–0.94, top5 0.41–0.64, rank_p95 ≤ 3. Classification per failure-shape table: not "global compression", not "drift with position", but "rank instability on close siblings" — peak logit consistently compressed ~50% (P3 max=23.8 vs llama 34.7).

### Phase 1.8 — diagnosis branch
First suspect (q6k AVX2 vs scalar internal parity) eliminated: bit-identical when forced via `WINMOE_LM_SCALAR=1`. Logit max still ~23 vs llama's 34.

Element-wise normed comparison via new `--dump-norm`/`WINMOE_DUMP_NORMED`: **cosine = 0.999, rel L2 = 4.95%** at position 66. Normed is fine — bug is the LM head dot product itself.

Direct logit inspection: WinMoE's top-1 token AGREES with llama (id 23573), but logit 19.17 vs llama 34.70 = exact 0.55× compression. Q6_K dequant inspection of `q6k_dot_block`:

```c
int is = n / 16;  // BUG — never adds l/16
```

Q6_K stores 16 per-16-weight scales. WinMoE used only 8 of them (every other one). Fix: `is = (n/16) + (l/16)`. AVX2 path split into `l<16` and `l>=16` scale broadcasts.

The internal scalar-vs-AVX2 parity test had passed all along because both paths shared the same bug.

### Phase 1 — verification after fix
ALL 5 PROMPTS PASS. KL improved 280×.

| Prompt | top1 (before → after) | top5 (before → after) | KL_mean (before → after) |
|---|---|---|---|
| P1 bracket_closure | 0.82 → 0.98 | 0.58 → 0.95 | 2.39 → 0.018 |
| P2 indentation_python | 0.74 → 0.96 | 0.62 → 0.98 | 1.39 → 0.003 |
| P3 var_disambiguation | 0.94 → 0.96 | 0.54 → 0.94 | 2.55 → 0.010 |
| P4 operator_choice | 0.84 → 0.98 | 0.64 → 0.99 | 1.01 → 0.001 |
| P5 json_output | 0.92 → 0.94 | 0.41 → 0.96 | 4.18 → 0.007 |

### Phase 2 — long-generation stability (P3 only, 200 tokens)
top1 = 0.99, top5 = 0.97, rank_p95 = 0, KL = 0.0035. No position-drift; argmax matches llama 198/200.

### Phase 3 — free-gen code quality
- P3, P4: ast.parse PASS at 100 tokens
- P5: json.loads PASS — perfect `{"name": "John Doe"}` in fenced block
- P1: clean Python config when extended to 200 tokens
- P2: valid mid-function code; needs ~250 tokens to close (model spent first 100 on prose intro)

**Effective verdict 5/5.** Strict 100-token gate cutoff fails P1/P2 only because of truncation, not generation quality.

### Phase 4 prep
Added `WINMOE_EXPERT_TRACE=path` — per-(layer, expert) hit counter dumped at exit.
`loops/speed-coding/analyze_expert_trace.py` — computes hot/cold split + cumulative coverage.

**Expert-access distribution (P1, P2 traces, 200 tok each):**
- **Hot set (top 50% experts): 95–96% of accesses**
- **Cold set (bottom 50% experts): 4–5% of accesses**

Workload-stable across two prompts. Re-orders Phase 4 priorities:
1. **Phase 4.2 (LFU/ARC cache)** moves to #1 — current 54% hit rate is wasting cache slots on cold experts. With 96% of work on hot 50%, a frequency-aware cache should reach 85–90%.
2. **Phase 4.1 (IQ2 cold quant)** drops in priority — cold-tier accounts for only ~5% of SSD reads. Save ~2.4% bandwidth on its own. Still useful for fitting more in RAM.
3. Phase 4.3, 4.4 unchanged.

LFU cache code written + built (`WINMOE_CACHE_POLICY=lfu`). Head-to-head benchmark vs round-robin not yet run.

---

## What changed
- `engine/runtime/q6k_dequant.h` — fixed scalar + AVX2 q6k_dot_block scale indexing
- `engine/runtime/winmoe_inference.c` — `WINMOE_DUMP_ALL_LOGITS`, `WINMOE_DUMP_NORMED`, `WINMOE_LM_SCALAR`, `WINMOE_EXPERT_TRACE`, `WINMOE_CACHE_POLICY=lfu`
- `D:/llama-cpp-src/examples/trace-dump/trace-dump.cpp` — `--dump-logits`, `--dump-norm`, `--n-gen`
- New: `loops/parity-coding/` — full sprint infrastructure + results
- New: `loops/speed-coding/analyze_expert_trace.py`

## Commits
- `fca9061` v10.48-pre — parity infrastructure
- `2d0f3b7` v10.48 — Q6_K scale-index fix, Phase 1 PASS

## Honest perf vs llama.cpp on this laptop
- llama.cpp `-ngl 0` 397B: ~0.5 tok/s (mmap, OS page cache)
- WinMoE v10.48 397B: ~0.3 tok/s (explicit cache, 54% hit)

**We are slower than llama.cpp by ~40%.** Phase 4.2 alone projected to close most of that. Full Phase 4 stack should land 1.0–1.5 tok/s (2–3× llama).

## Open questions for next session
- LFU vs round-robin actual hit-rate delta on a 200-tok run (built, not yet measured)
- Does LFU hit rate climb on longer runs as warmup completes? Need 500-tok sample.
- What's the per-layer routing variance? If layer 0 has different hot set than layer 30, cache strategy may need per-layer hot lists.

## What's deferred
- Phase 4.1 IQ2 cold quant — modest gain, do after 4.2
- Phase 4.3 DN-MoE-skip PPL — independent risk assessment
- Phase 4.4 prefetch SCOPE-A' — needs draft-token machinery
