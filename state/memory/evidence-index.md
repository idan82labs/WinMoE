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
