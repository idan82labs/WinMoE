# Professor Questions V2 — Based on Empirical Results

These questions are informed by measured data from running Qwen3.5-397B-A17B on
consumer Windows hardware (RTX 3070 8GB, 40GB DDR4, Samsung 980 NVMe 2.1 GB/s).

---

## Q1: NVMe Queue Contention Under Concurrent mmap + Explicit I/O

**Measured**: Background explicit I/O (prefetching expert pages) competing with
mmap-based inference reads **degrades** total throughput rather than improving it.
The same phenomenon was observed on Apple Silicon (Flash-MoE).

Standalone explicit I/O achieves 2,200 MB/s. mmap achieves 586 MB/s.
Running both simultaneously achieves worse than either alone.

**Question**: Is this a fundamental queueing theory result? On a single NVMe
device with one command queue (or limited queue count), does adding a second
I/O stream to "help" the first always degrade total throughput when both
streams access the same physical blocks?

Formally: Let the NVMe device have service rate mu with K command queue slots.
Stream A (mmap faults) generates arrivals at rate lambda_A. Stream B (prefetch)
adds lambda_B. Under what conditions does adding B cause:
```
throughput(A+B) < throughput(A alone)
```

Is this related to the **processor sharing** or **discriminatory scheduling**
models? Is there a critical load threshold beyond which "helpful" prefetch
becomes destructive?

**Why this matters**: If this is fundamental, then the ONLY way to capture the
3.7x explicit I/O speedup is to **replace** mmap, never to supplement it.
This determines whether we need full llama.cpp source modification or can use
a wrapper approach.

---

## Q2: Optimal Tier Sizing Under Non-Stationary Expert Routing

**Measured**: Expert routing Zipf exponent varies by layer (0.19-0.40 per layer
in Qwen3-30B, 0.46 mean from OLMoE real inference). Consecutive-token overlap
is 37.5%. Working set grows sub-linearly with sequence length but eventually
covers 96% of experts by 1000 tokens.

The simulator shows:
- VRAM tier (1000 blocks): 64% hit rate
- RAM tier (5000 blocks): 31% hit rate
- SSD tier: 5% miss rate
- Combined: 95% hit rate

**Question**: Given a **non-stationary** expert access process where:
- Short-term routing depends on input content (task-specific expert clusters)
- Long-term routing converges to a stationary distribution
- The transition between task-specific and stationary regimes happens over ~50-200 tokens

What is the **optimal dynamic tier sizing** strategy? Specifically:

1. Should the VRAM-RAM boundary be static (based on long-term frequency) or
   adaptive (tracking recent access patterns)?

2. The professor's V2 spec framework defines:
   ```
   max_{C_s + C_a = C_V} V_static(C_s) + V_adaptive(C_a; C_s)
   ```
   With our measured parameters (Zipf=0.46, overlap=37.5%, 512 experts/layer),
   what is the **optimal C_s / C_a split** for the VRAM tier?

3. Is there a **phase transition** in the optimal policy as sequence length
   increases? (Short sequences → adaptive wins; long sequences → static wins?)

---

## Q3: The Compute-I/O Crossover Under Pipeline Overlap

**Measured**: Our system has two regimes:
- **I/O-bound** (current): 588 ms/token, ~400ms expert I/O, ~200ms compute
- **Projected with explicit I/O**: 282 ms expert I/O, ~200ms compute

With pipeline overlap (read layer N+1 while computing layer N):
```
t_token = max(t_compute, t_io) per layer, summed across 60 layers
```

**But** the NVMe contention finding (Q1) suggests that overlap may not be
achievable — if reading and computing simultaneously degrades both.

**Question**: Under the **causal pipeline model** where overlap IS possible
(discrete GPU + separate NVMe PCIe lanes), derive the **exact crossover point**:

At what expert-block service rate does the system transition from I/O-bound
to compute-bound? Express as:
```
B_crossover = f(K, expert_size, T_c, num_layers)
```

With our parameters:
- K=3, expert_size=3.3MB, T_c=7.12ms/layer, 60 layers
- What SSD bandwidth makes us compute-bound?
- Is our Samsung 980 (2.1 GB/s) above or below crossover?
- Is a Gen4 NVMe (6.5 GB/s) above or below crossover?

The integrated benchmark suggests all explicit I/O methods converge to
2.34 tok/s with overlap — is this the theoretical compute-bound ceiling?

---

## Q4: Information-Theoretic Limits on Expert Prediction with Measured Overlap

**Measured**: Consecutive-token expert overlap is 37.5% (from OLMoE real inference).
This means 3/8 experts in layer L at token t are the same as at token t-1.

Flash-MoE found temporal prediction had only 25% accuracy and was harmful.
Our prefetch wrapper confirmed: external prediction doesn't help when it
competes for the same I/O channel.

**Question**: If pipeline overlap IS achievable (after replacing mmap):

1. With 37.5% known overlap, what is the **Bayes-optimal predictor** for the
   remaining 62.5%? What features would it need?
   - Current hidden state (available before routing)
   - Previous token's expert set (available)
   - Layer-specific routing statistics (pre-computed)

2. What **recall rate R** at what **over-prefetch budget u** is needed to make
   prediction worthwhile? The V2 spec defines:
   ```
   t_layer(Delta) ≈ T_c + (1 - R(u_Delta)) * t_f_real
   ```
   With our measured T_c=7.12ms and t_f_real=5.12ms (mmap) or 1.67ms (explicit),
   what R threshold makes prediction beat static caching?

3. Is there a **fundamental limit** on prediction accuracy given the measured
   routing statistics? If the mutual information I(E_t; E_{t-1}) is low
   (implied by 37.5% overlap), prediction may be mathematically bounded below
   useful thresholds.

---

## Q5: Cache Cliff Topology for MoE with Measured Service-Demand Distribution

**Measured**: The simulator's rho(C) curve shows:
- rho(5000 blocks) = 1.38 (K=10)
- rho(5000 blocks) = 0.62 (K=3)
- rho drops roughly linearly with cache size in the 3000-8000 block range

The V2 spec asks for:
```
C_q* = inf{C : Q_q[S_t(C)] <= T_c}  for q = 0.95, 0.99
```

**Question**:

1. Is the cache cliff for MoE expert loading a **smooth transition** or a
   **sharp phase transition**? The DirectStorage LLM project found that reducing
   cache by 25% caused >1000 faults/token (catastrophic). Our simulator shows
   smoother degradation. Which is correct and why?

2. For our measured Zipf=0.46 distribution on 512 experts/layer:
   - What is the **analytical form** of the miss rate as a function of cache size?
   - Is it M(C) ≈ (1 - C/N)^alpha for some alpha, or does the non-IID
     temporal structure change the functional form?
   - What is the **derivative dM/dC** at the operating point (C=5000)?
     This tells us the marginal value of one more cached block.

3. With measured service times (SSD: beta=42us + x/2.1GB/s, PCIe: 11us + x/19GB/s):
   what is the **minimum cache size** for 99th percentile service demand to fit
   within compute slack (7.12ms/layer)?

---

## Q6: Error Propagation in Sequential MoE Inference with Reduced K

**Measured**: At IQ2_XXS quantization, K=3 and K=5 produce indistinguishable
output quality on our test prompts. The agent hypothesized that 2-bit quantization
noise dominates expert-count noise.

**Question**: Formalize this:

1. Model the per-token hidden state as h_t = f_K(h_{t-1}) + epsilon_quant + epsilon_K
   where epsilon_quant is quantization noise and epsilon_K is expert-dropping noise.

   Under what conditions does Var(epsilon_quant) >> Var(epsilon_K), making K
   reduction "free" from a quality perspective?

2. For IQ2_XXS (~2-bit), what is the expected quantization noise floor?
   At Q4_K_M (~4-bit), the noise is ~4x lower. Does K=3 vs K=5 become
   distinguishable at Q4?

3. Is there a **critical quantization precision** below which K reduction is
   always free, and above which it matters? This would define the optimal
   (quantization, K) Pareto frontier for a given hardware budget.

4. Over L=60 sequential layers, does the error compound? The Lipschitz
   constant question from V1 remains: under what conditions on the layer
   function's Lipschitz constant L_f does the compounding error
   ||h_t - h_t*|| remain bounded vs diverge over a 500-token generation?

---

## Q7: Fundamental Throughput Limit for SSD-Streamed MoE on Discrete GPU

**Measured**: Every optimization path converges to the same ceiling:
- Explicit I/O with overlap: 2.34 tok/s
- All cache configurations with overlap: 2.34 tok/s
- This appears to be the **compute-bound ceiling** at K=3, ngl=8

**Question**: Derive the **closed-form throughput ceiling** for SSD-streamed
MoE inference on a discrete-GPU Windows system:

```
TPS_max = f(B_ssd, B_pcie, T_c, K, expert_size, N_layers, C_vram, C_ram, N_experts, zipf_s)
```

With our parameters:
- B_ssd = 2.1 GB/s, B_pcie = 19 GB/s
- T_c = 7.12 ms/layer, K = 3, expert_size = 3.3 MB
- N_layers = 60, C_vram = 1000 blocks, C_ram = 5000 blocks
- N_experts = 512/layer, zipf_s = 0.46

What does this formula predict?

More importantly: **what hardware upgrade gives the highest marginal return?**
- More RAM (→ higher cache hit rate, lower miss volume)?
- Faster SSD (→ lower miss service time)?
- More VRAM (→ more hot experts, less PCIe traffic)?
- Better GPU (→ lower T_c, higher compute ceiling)?

The answer determines whether to buy a Gen4 NVMe, more RAM, or a better GPU.

---

## Priority Order

1. **Q1** (NVMe contention) — determines the integration architecture
2. **Q3** (compute-I/O crossover) — determines the performance ceiling
3. **Q7** (throughput formula) — determines hardware upgrade ROI
4. **Q5** (cache cliff) — determines tier sizing
5. **Q2** (dynamic tier sizing) — determines cache policy
6. **Q6** (K vs quantization) — determines quality frontier
7. **Q4** (prediction limits) — determines if prediction is worth pursuing
