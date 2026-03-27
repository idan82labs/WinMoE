# Session 001 — Initial Execution

Date: 2026-03-25

## What was done

### Phase -1 (complete)
- Unpacked project scaffold from zip
- Verified all directories and files

### Phase 0 — Systems Baseline (substantially complete)
1. **Hardware profiled**: RTX 3070 Laptop 8GB, 40GB RAM, Samsung 980 1TB + Micron 2210 512GB NVMe
2. **PyTorch 2.11+cu128 installed and verified**
3. **I/O benchmark harness built and run** (`io_benchmark.py`):
   - Explicit unbuffered reads: 2.1-2.9 GB/s, beta≈42µs/req
   - Explicit buffered: 6.8 GB/s (OS cache, misleading)
   - mmap: ~1.1 GB/s (page cache illusion)
4. **PCIe benchmark built and run** (`pcie_benchmark.py`):
   - Pinned H2D: 18-19 GB/s, 11µs/transfer
   - Pageable H2D: 6-8 GB/s
5. **Baseline I/O path selected**: explicit unbuffered reads (FILE_FLAG_NO_BUFFERING) + pinned memory staging
6. **First affine timing fits recorded** for SSD and PCIe stages
7. **io-path-loop**: 3 trials (baseline + 2 rejected candidates)
8. **timing-fit-loop**: baseline established

### Phase 1 — Trace Acquisition (started)
1. **Model architecture researched**: Qwen3-235B, Qwen3.5-397B configs obtained from HuggingFace
2. **Target confirmed**: Qwen3.5-397B-A17B at Q4 (6.29 MB/expert, K=10, 60 layers, 512 experts)
3. **Trace capture plan written**: HF forward hooks on Qwen3-30B-A3B (18 GB Q4)
4. **Trace capture tool written** (`capture_traces.py`) — ready to run when model is downloaded
5. **transformers + bitsandbytes installed**

### Phase 2 — Simulator (built early, validated)
1. **Cache simulator built** (`moe_cache_simulator.py`):
   - Static LFU, LRU, hybrid policies
   - Oracle (Belady's MIN) framework
   - Service-demand distribution, load factor, critical cache sizing
2. **Comprehensive sweeps run** (`run_sweep.py`):
   - RAM capacity sweep: need ~63 GB at zipf=1.2, ~47 GB marginal
   - Zipf sensitivity: viability threshold at zipf_s≈1.3-1.4
   - Token length sensitivity: short prompts viable, long prompts not

### Phase 4 — K_l Analysis (built early)
1. **K reduction analysis built** (`k_reduction_analysis.py`):
   - K=10→7 makes this hardware viable (rho 1.38→0.87)
   - K×RAM viability matrix computed
   - K_l confirmed as most impactful lever

## Key decisions made
- Baseline I/O: explicit unbuffered reads selected, mmap rejected
- Primary target: Qwen3.5-397B-A17B at Q4
- Pinned memory mandatory for RAM→GPU
- Static LFU confirmed as best policy family

## What remains

### For Gate 0A completion
- [ ] Correctness reference path (needs real model weights)

### For Phase 1
- [ ] Download Qwen3-30B-A3B and capture real routing traces
- [ ] Calibrate synthetic trace parameters against real data

### For Phase 2+
- [ ] Run simulator with real traces
- [ ] Oracle comparison
- [ ] VRAM split optimization
- [ ] Service-demand quantile analysis
- [ ] KV-compression scenarios

## Artifacts produced
- `work/baseline/io_benchmark.py`
- `work/baseline/pcie_benchmark.py`
- `work/baseline/results_ssd.tsv`, `results_pcie.tsv`
- `work/baseline/notes/hardware-profile.md`
- `work/baseline/notes/model-architecture-reference.md`
- `work/baseline/notes/timing-analysis-001.md`
- `work/sim/moe_cache_simulator.py`
- `work/sim/run_sweep.py`
- `work/sim/k_reduction_analysis.py`
- `work/sim/results/analysis-001.md`
- `work/sim/results/*.json` (sim, ram_sweep, zipf_sweep, k_reduction)
- `work/traces/capture_traces.py`
- `work/traces/trace-acquisition-plan.md`
- `work/timings/fits/affine_fit_ssd.json`, `affine_fit_pcie.json`
