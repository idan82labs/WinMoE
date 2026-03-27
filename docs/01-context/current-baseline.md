# Current Baseline

## Settled first-order corrections
- The relevant per-layer miss payload is the active expert set, not a single expert block.
- The conservative Windows decode model is causal unless prediction creates real lead time.
- Zero-cache Windows on the target class of consumer NVMe is not the main viability story.
- The first-order question is whether RAM-first caching suppresses SSD misses enough.
- Timing must be calibrated with startup and per-request overhead terms before strong tok/s conclusions are drawn.

## Systems-bringup corrections
- The practical starting point is not `mmap` plus later optimization; it is explicit offset-based reads, explicit staging, and slab-oriented experiments.
- Page-cache pollution and write-back behavior can materially distort local-MoE experiments.
- Correctness bugs in routing, shared-expert handling, or cache logic can masquerade as systems bottlenecks.

## New cross-lever
- KV-cache compression should be treated as a memory-budget lever that may free RAM/VRAM for expert caching, but it does not replace expert-streaming analysis.

## Consequence
The V2 program is now a systems-baseline + trace-driven, calibrated systems study.
