# External Signals — Kimi Local Bring-up and KV Compression

## Why this file exists
Capture two external signals that changed the recommended starting point for the V2 program.

## Signal A — Local giant-MoE bring-up lessons from Kimi-style worklogs
Observed lesson pattern:
- `mmap` / paging style access can fail badly via page thrashing
- explicit read primitives and page-cache control can produce large early gains
- correctness fixes can matter as much as throughput fixes during bring-up
- early wins are often boring systems hygiene, not advanced scheduling or prediction

## Program implications
- Add a systems-baseline phase before broad trace/simulator work
- Add a hard correctness gate before performance claims
- Compare cached vs uncached explicit file I/O early
- Prefer contiguous slab reads and explicit staging over `mmap` as the default baseline

## Signal B — TurboQuant / QJL / PolarQuant line of work
Observed lesson pattern:
- KV-cache compression may materially shrink the memory budget consumed by long-context inference
- recovered memory can potentially be reallocated to expert caches or staging buffers
- this changes feasible VRAM/RAM envelopes without directly solving expert streaming

## Program implications
- add a KV-compression scenario workstream
- feed KV-compression scenarios into tier-sizing and simulator sweeps
- do not treat KV compression as a substitute for trace-driven expert-cache analysis
