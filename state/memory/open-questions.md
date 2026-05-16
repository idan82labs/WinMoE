# Open Questions

## Answered
- ~~What primitive I/O path should be the default baseline?~~ → **Explicit unbuffered reads (FILE_FLAG_NO_BUFFERING) + pinned staging**
- ~~How large are startup and per-request overhead terms?~~ → **SSD: beta≈42µs, B≈2.1 GB/s; PCIe: 11µs, 19 GB/s**
- ~~Does layerwise K_l buy more than predictor work?~~ → **Yes. K=10→7 shifts rho from 1.38→0.87 (synthetic). Most impactful near-term lever.**

## Still open — CRITICAL
- ~~**What is the real routing concentration (Zipf exponent) for Qwen3.5-397B?**~~ → **Top 50% of experts absorb 95–96% of accesses on 200-tok code prompts** (`phase4_expert_trace_p{1,2}.tsv`, May 2026). Workload-stable across two prompts. Implies strong skew, comfortably above zipf_s≈1.3 viability bar.
- Does LFU cache eviction (built but not yet measured) close the gap from 54% → 85%+ hit rate predicted by the skew? Head-to-head benchmark queued for next session.
- Per-layer routing variance: do all 45 MoE layers share the same hot-set, or do different layers concentrate on different experts? Affects whether the cache should pin globally or per-layer.

## Still open — important but not blocking
- Can routing traces be captured from Qwen3-30B-A3B with acceptable distortion? (Tool ready, model not yet downloaded)
- How stable are the top sets at real VRAM and RAM cutoffs? (Needs real traces)

## Still open — important but not blocking
- How large is the static vs recency vs oracle gap as a function of capacity? (Oracle sim ready but slow)
- How much memory budget can realistic KV compression scenarios return to expert caching?
- After the cached baseline exists, is there enough residual SSD/control-path fraction for DirectStorage to matter?
