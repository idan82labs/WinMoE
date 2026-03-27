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
