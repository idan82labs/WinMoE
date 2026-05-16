# Evidence Index

## Reserved buckets
- systems baseline / correctness
- trace acquisition
- timing calibration
- simulator outputs
- tier sizing / cache cliff
- KV-compression scenarios
- `K_l` scenarios
- prediction evaluation
- DirectStorage evaluation

## Collected evidence

### Systems baseline / correctness
- `work/baseline/notes/hardware-profile.md` — measured system specs (RTX 3070 8GB, 40GB RAM, Samsung 980 1TB + Micron 2210 512GB)
- `work/baseline/notes/model-architecture-reference.md` — Qwen3/3.5 architecture details from HuggingFace configs
- `work/baseline/notes/timing-analysis-001.md` — first SSD + PCIe timing analysis
- `work/baseline/io_benchmark.py` — SSD I/O benchmark harness (explicit unbuf vs buffered vs mmap)
- `work/baseline/pcie_benchmark.py` — PCIe RAM→GPU benchmark harness (pinned vs pageable)
- `work/baseline/results_ssd.tsv` — SSD benchmark raw data
- `work/baseline/results_pcie.tsv` — PCIe benchmark raw data
- `loops/io-path-loop/results.tsv` — I/O path bakeoff: 3 trials (baseline + 2 rejected candidates)

### Parity sprint (v10.48)
- `loops/parity-coding/results.tsv` — 5 prompts × {before-fix, after-fix} parity gate results
- `loops/parity-coding/results/0[1-5]_*/` — per-prompt PNG plots (top1, top5, rank, delta) + metrics.json
- `loops/parity-coding/results/phase2_03_var_disambiguation/` — 200-tok teacher-forced parity for prompt 3
- `loops/parity-coding/results/phase3/phase3_results.json` — free-gen ast.parse/json.loads outcomes per prompt
- `loops/parity-coding/harness.py` — top-k/rank/delta/KL metrics + drift plots
- `loops/parity-coding/run_parity.py` — orchestrator (5 prompts × {llama, winmoe} × harness)
- `loops/parity-coding/run_phase3.py` — free-gen driver + ast/json parser
- `state/progress/session-log/002-q6k-fix-parity-sprint.md` — full session writeup

### Expert-access profile (Phase 4 prep)
- `loops/speed-coding/analyze_expert_trace.py` — hot/cold split analyzer
- Two long-gen traces (P1, P2, 200 tok each) show 95–96% accesses concentrate in top 50% of experts — workload-stable. Output dumped to `C:/Users/idant/phase4_expert_trace_p{1,2}.tsv` (not in repo).

### Timing calibration
- `work/timings/fits/affine_fit_ssd.json` — first SSD affine fit (alpha≈0, beta≈42µs, B≈2118 MB/s)
- `work/timings/fits/affine_fit_pcie.json` — PCIe fit (overhead≈11µs, B≈19 GB/s pinned)

### Simulator outputs
- `work/sim/moe_cache_simulator.py` — trace-driven cache simulator (static LFU, LRU, hybrid, oracle)
- `work/sim/run_sweep.py` — comprehensive parameter sweep tool
- `work/sim/results/analysis-001.md` — first viability assessment with synthetic traces
- `work/sim/results/sim_results.json` — raw simulator outputs (100 tokens)
- `work/sim/results/ram_sweep.json` — RAM capacity sweep (500 tokens)
- `work/sim/results/zipf_sweep.json` — Zipf concentration sensitivity

### K_l scenarios
- `work/sim/k_reduction_analysis.py` — K reduction and K×RAM viability matrix tool
- `work/sim/results/k_reduction.json` — K sweep raw data

### Key simulator findings (synthetic traces, static LFU)
- Viability requires zipf_s > ~1.3 at 32 GB RAM, or ~63 GB RAM at zipf_s=1.2
- This hardware (40 GB RAM, 8 GB VRAM) is at the boundary
- Real routing concentration is the decisive unknown
- Static LFU >> LRU; adaptive policies offer no advantage for RAM tier

### Key K_l findings
- K=10→7 makes this hardware viable (rho 1.38→0.87) while retaining 88.5% gate mass
- K×RAM viability matrix: K=6 viable at 19GB RAM; K=8 needs 47GB; K=10 needs 63GB
- K_l reduction is more impactful than prediction or DirectStorage
