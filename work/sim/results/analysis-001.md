# Simulator Analysis 001 — First Viability Assessment

Date: 2026-03-25
Model: Qwen3.5-397B-A17B at Q4 (6.29 MB/expert, K=10, 60 layers, 512 experts/layer)
Hardware: RTX 3070 8GB VRAM, 40GB RAM, Samsung 980 NVMe

## Setup
- Synthetic Zipf-distributed routing traces
- Static LFU caching policy (best performing)
- VRAM capacity: 1000 blocks (~6.3 GB, realistic for 8GB GPU after overhead)
- Timing: SSD beta=42µs, B=2.1 GB/s; PCIe beta=11µs, B=19 GB/s; T_c=5ms

## Key Result 1: RAM capacity is the critical variable

| RAM (blocks) | RAM (GB) | SSD miss% | Mean svc (ms) | rho | Status |
|-------------|----------|-----------|----------------|------|--------|
| 5,000 | 31.5 | 17.7% | 6.90 | 1.38 | MARGINAL |
| 7,500 | 47.2 | 12.8% | 5.45 | 1.09 | MARGINAL |
| 10,000 | 62.9 | 9.4% | 4.39 | 0.88 | **VIABLE** |
| 12,000 | 75.5 | 7.3% | 3.77 | 0.75 | **VIABLE** |

**At zipf_s=1.2, need ~63 GB RAM for viability. This hardware (40 GB) is marginal.**

## Key Result 2: Expert concentration (zipf_s) determines viability

| zipf_s | SSD miss% | Mean svc (ms) | rho | Status |
|--------|-----------|----------------|------|--------|
| 0.8 | 38.2% | 13.87 | 2.78 | NOT VIABLE |
| 1.0 | 26.6% | 9.95 | 1.99 | NOT VIABLE |
| 1.2 | 17.7% | 6.90 | 1.38 | MARGINAL |
| **1.5** | **8.8%** | **3.78** | **0.76** | **VIABLE** |
| 1.8 | 4.1% | 2.05 | 0.41 | VIABLE |
| 2.0 | 2.3% | 1.35 | 0.27 | VIABLE |

**Viability threshold at ~zipf_s=1.3-1.4.** Real Qwen3 routing shows Gini=0.30 and 14.3x max/min ratio — this likely corresponds to zipf_s ≈ 1.0-1.3.

## Key Result 3: Longer sequences worsen viability

| Tokens | Unique experts | SSD miss% | rho |
|--------|---------------|-----------|------|
| 50 | 9,338 | 11.1% | 0.978 |
| 100 | 13,965 | 14.3% | 1.170 |
| 500 | 26,265 | 17.7% | 1.380 |
| 1000 | 29,558 | 18.0% | 1.401 |

Working set grows with sequence length but plateaus. Short prompts (<50 tokens) are viable on this hardware; long prompts are not.

## Conclusions

1. **The system is at the boundary of viability** on this specific hardware (40 GB RAM).
2. **Real routing concentration is the decisive unknown.** If zipf_s > 1.4, the system works with 32 GB RAM. If zipf_s < 1.2, it doesn't work even with 64 GB.
3. **Static LFU is clearly the right policy family.** LRU and hybrid offer no advantage.
4. **Layerwise K_l reduction is the most promising lever** — reducing K from 10 to 6-7 would proportionally reduce SSD miss payload by 30-40%.
5. **A PCIe 4.0 NVMe (6-7 GB/s vs 2.1 GB/s) would shift viability significantly** — ~3x faster SSD would make rho ~0.5 at zipf_s=1.2.
