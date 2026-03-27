# Open Questions

## Answered
- ~~What primitive I/O path should be the default baseline?~~ → **Explicit unbuffered reads (FILE_FLAG_NO_BUFFERING) + pinned staging**
- ~~How large are startup and per-request overhead terms?~~ → **SSD: beta≈42µs, B≈2.1 GB/s; PCIe: 11µs, 19 GB/s**
- ~~Does layerwise K_l buy more than predictor work?~~ → **Yes. K=10→7 shifts rho from 1.38→0.87 (synthetic). Most impactful near-term lever.**

## Still open — CRITICAL
- **What is the real routing concentration (Zipf exponent) for Qwen3.5-397B?** This is the single most important remaining unknown. Literature suggests zipf_s ≈ 1.0-1.3. If ≥ 1.3, this hardware is viable; if < 1.2, it is not without K_l reduction.
- Can routing traces be captured from Qwen3-30B-A3B with acceptable distortion? (Tool ready, model not yet downloaded)
- How stable are the top sets at real VRAM and RAM cutoffs? (Needs real traces)

## Still open — important but not blocking
- How large is the static vs recency vs oracle gap as a function of capacity? (Oracle sim ready but slow)
- How much memory budget can realistic KV compression scenarios return to expert caching?
- After the cached baseline exists, is there enough residual SSD/control-path fraction for DirectStorage to matter?
