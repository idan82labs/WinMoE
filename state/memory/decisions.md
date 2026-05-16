# Decisions

- Treat V2 as an execution program, not a writing program.
- Keep prediction and DirectStorage gated behind cached-baseline evidence.
- Add a systems-baseline phase before broad trace/simulator work.
- Treat explicit offset-based reads and slab-oriented experiments as the default starting point.
- Add KV-compression scenarios as an explicit workstream affecting tier budgets.
- [2026-03-25] **Baseline I/O path selected: explicit unbuffered reads (FILE_FLAG_NO_BUFFERING) via Win32 CreateFileW/ReadFile.** Explicit buffered rejected (OS cache illusion). mmap rejected (page thrashing for >>RAM, no async control, 4KB fault granularity). Measured: beta≈42µs/req, B≈2.1 GB/s on Samsung 980 NVMe.
- [2026-03-25] **Pinned memory mandatory for RAM→GPU transfers.** Pageable gives only 6-8 GB/s; pinned gives 18-19 GB/s on RTX 3070 Laptop PCIe Gen 4.
- [2026-03-25] **Primary target model: Qwen3.5-397B-A17B at Q4 quantization.** This matches the spec's K=10, ~6.75 MB/expert, D≈67.5 MB/layer. 60 layers, 512 experts/layer, 1 shared expert.
- [2026-03-25] **Real routing traces captured from OLMoE-1B-7B.** Zipf=0.456, Gini=0.396, consecutive overlap=37.5% (3/8 experts shared). Max/min ratio 50-61x. Much higher concentration than gate-weight-only routing (0.275) and higher than Qwen3-30B reported Gini (0.30).
- [2026-03-25] **External validation: iPhone 17 Pro runs Qwen3.5-397B at 0.6 tok/s.** Confirms consumer hardware MoE streaming is viable with Apple Silicon unified memory + fast NVMe.
- [2026-05-16] **Correctness sprint precedes speed sprint.** Freeze graph (K=10, no sampling), do teacher-forced full-logit parity vs llama.cpp BEFORE any throughput work. Rationale: "RMS matches" ≠ "logits match" — scalar RMS could agree even if components diverge. We caught the Q6_K scale-index bug only because we ran position-by-position cosine of normed + element-wise logit comparison.
- [2026-05-16] **Phase 1 gate (5 prompts × 50 tok teacher-forced): top1≥0.9, top5≥0.7, rank_p95≤3, delta_ratio_p90≤1.5.** Passed by all 5 after the Q6_K fix. Used as the canonical correctness checkpoint.
- [2026-05-16] **Phase 4 priority re-ordered after expert-trace analysis.** Top 50% experts handle 95–96% of accesses (stable across prompts). LFU/ARC cache (Phase 4.2) is the single biggest throughput lever; IQ2 cold quant (Phase 4.1) drops to #2 because cold tier only carries ~5% of SSD bandwidth.
- [2026-05-17] **Cache eviction Phase 4.2 deprioritized after measurement.** LFU (raw) and LFU-DA (with seeded inserts) both regress tok/s vs round-robin despite higher hit rate (52% → 58.5%). Two reasons: (1) round-robin's rotating evictions exploit Windows file-cache as a "warm tier" — LFU pushes long-tail misses to disk-cold path where each miss costs ~10× more; (2) expert FFN compute (~1700ms/token) dominates IO; eviction can't reduce compute. Pivot to compute-side levers: more GPU experts, prefetch overlap, DN-MoE-skip.
- [2026-05-16] **MacMoE deferred until M5 Max 128GB arrives.** Unified memory + 500 GB/s BW makes the SSD-streaming architecture inappropriate for that platform; MLX or llama.cpp/Metal is the right engine there. The DN per-head KV indexing fix + the parity harness do transfer.
