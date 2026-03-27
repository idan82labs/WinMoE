# Analysis 003 — Final Viability Assessment with Real Routing Data

Date: 2026-03-25
Based on: Real OLMoE-1B-7B routing traces (Zipf=0.456, Gini=0.396, overlap=37.5%)

## iPhone 17 Pro reference: 0.6 tok/s on Qwen3.5-397B

## Viability Matrix: Qwen3.5-397B (VRAM=1000, RAM=5000=31GB, Zipf=0.456)

| Target tok/s | Samsung 980 (2.1 GB/s) | PCIe 3.0 (3.5 GB/s) | PCIe 4.0 (6.5 GB/s) |
|-------------|------------------------|----------------------|----------------------|
| 0.3 | rho=0.37 **VIABLE** | rho=0.25 **VIABLE** | rho=0.16 **VIABLE** |
| 0.5 | rho=0.62 **VIABLE** | rho=0.41 **VIABLE** | rho=0.27 **VIABLE** |
| 1.0 | rho=1.24 MARGINAL | rho=0.82 **VIABLE** | rho=0.53 **VIABLE** |
| 2.0 | rho=2.49 NOT VIABLE | rho=1.65 NOT VIABLE | rho=1.06 MARGINAL |
| 5.0 | rho=6.22 NOT VIABLE | rho=4.11 NOT VIABLE | rho=2.66 NOT VIABLE |

## K Reduction at 1.0 tok/s, Samsung 980 (2.1 GB/s), RAM=31GB

| K | rho | Status |
|---|-----|--------|
| 4 | 0.47 | **VIABLE** |
| 5 | 0.59 | **VIABLE** |
| 6 | 0.72 | **VIABLE** |
| 7 | 0.85 | **VIABLE** |
| 8 | 0.98 | **VIABLE (barely)** |
| 10 | 1.23 | MARGINAL |

## Key Conclusions

1. **This hardware (Samsung 980 + 40 GB RAM) can run 397B at 0.5 tok/s** with K=10 static LFU.
2. **At 1.0 tok/s, K reduction to K<=8 is needed** on Samsung 980 (2.1 GB/s).
3. **A PCIe 4.0 NVMe (6.5 GB/s) makes 1 tok/s viable** at K=10 without K reduction.
4. **2+ tok/s requires PCIe 4.0 NVMe + K<=6** or 64+ GB RAM.
5. **The iPhone 17 Pro at 0.6 tok/s is consistent** with our model at rho=0.37-0.62.

## Verdict: **GO** at 0.5 tok/s, **CONDITIONAL GO** at 1.0 tok/s

The system is viable for interactive use (0.5-1.0 tok/s) on consumer Windows hardware.
The primary lever for higher throughput is faster NVMe, not more RAM.
