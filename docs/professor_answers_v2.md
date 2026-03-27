# Professor Answers V2 — Summary of Key Results

## Q1: NVMe Contention — ARCHITECTURE DECISION
**Answer**: The phenomenon IS fundamental enough to drive architecture.

On one SSD, speculative prefetch helps only if it **replaces** faults, not coexists.
The right design: let explicit I/O **own** the expert data plane, demote mmap to fallback.

Key formula (M/G/1-PS model):
```
R_A ≈ s_A / (1 - rho_A - rho_B)
```
Adding stream B always increases A's response time unless B eliminates enough future faults.

**Implication**: Replace mmap. Do not supplement it.

## Q3: Compute-I/O Crossover — CEILING CONFIRMED
**Answer**: Cold-path crossover is **1.39 GB/s**.

```
B_crossover = K × S_e / T_c = 3 × 3.3MB / 7.12ms ≈ 1390 MB/s
```

- Samsung 980 explicit (2.1 GB/s): **ABOVE crossover** → compute-bound
- mmap (586 MB/s): **BELOW crossover** → I/O-bound
- Gen4 NVMe (6.5 GB/s): above, but no additional benefit for batch-1

**2.34 tok/s IS the compute ceiling** under overlap assumptions.
T_comp = 60 × 7.12 = 427.2 ms → 1/0.4272 = 2.34 tok/s.

**Critical insight**: With 95% cache hit rate (m≈0.05), crossover drops to 0.069 GB/s.
Storage bandwidth is NOT the bottleneck — the fault path / scheduling path is.

## Q7: Throughput Formula — HARDWARE ROI
**Answer**: Best marginal return ordering:
1. **Replace mmap with explicit async path** (software, biggest gain)
2. **Improve GPU compute / lower T_c** (more VRAM if it raises h_v)
3. **Faster SSD** (low ROI until compute gets faster)

A Gen4 NVMe does NOT help batch-1 decode after replacing mmap.

## Q5: Cache Cliff — SMOOTH, NOT PHASE TRANSITION
**Answer**: No true phase transition. Miss curve is smooth:

```
M_static(C) ≈ 1 - (C^0.54 - 1) / (512^0.54 - 1)
```

The "cliff" is a smooth miss curve crossing a hard max() bottleneck in layer time.

Key cache sizes (per layer, 512 experts):
- K=3 on explicit (2.1 GB/s): need C ≈ 192 (h ≈ 58%)
- K=10 on mmap (586 MB/s): need C ≈ 401 (h ≈ 87%)
- K=3 on mmap at p99: need C ≈ 459 (h ≈ 94%)

**Strongest evidence mmap is wrong**: K=3 at p99 on mmap needs 459/512 experts cached (90%!).

## Q2: Tier Sizing — HYBRID, ADAPTIVE-TO-STATIC
**Answer**: Use hybrid C_s (static) + C_a (adaptive).

- Higher-skew layers get more persistent VRAM (equal-marginal allocation)
- Early decode: larger adaptive share
- Later decode: larger static share
- Marginal value of recency decays after footprint saturates (~1000 tokens)

## Q6: K vs Quantization — CONFIRMED FREE AT IQ2
**Answer**: At IQ2_XXS, Var(epsilon_quant) >> Var(epsilon_K). K reduction is nearly free.

Critical precision threshold:
```
sigma_q^2 ≈ sigma_E^2 × (sum g_j^2 for j=4,5) / (sum g_j^2 for j=1,2,3)
```

Bounded propagation requires gate margin g_K - g_{K+1} > quantization perturbation.

## Q4: Prediction — WEAK DIRECTION
**Answer**: Prediction unlikely to beat good hotset + recency cache.

- 37.5% overlap is the easy part (just carry forward)
- Remaining 62.5% collapses toward static conditional prior = caching
- Label entropy is ~24.4 bits per expert set → Fano says high Bayes error
- Prediction fails on timeliness and precision, not bandwidth

## Bottom Line

**The single most important next step**: Replace mmap with explicit I/O inside the
inference loop. This is the software architecture change that captures the 3.7x speedup.

After that, improve GPU compute (lower T_c). A faster SSD is low ROI.
