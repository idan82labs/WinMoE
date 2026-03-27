# V2 Charter

## Objective
Produce a final V2 research report that gives a hard answer on Flash-MoE Windows feasibility under a RAM-first caching design.

## Core question
Can a practical tiered cache on the intended Windows hardware keep residual miss-service demand below compute slack often enough to make decode viable?

## Updated framing
The program now has a required **systems-baseline phase** before full trace-driven policy work. Recent external signals suggest that local giant-MoE bring-up is often bottlenecked first by the primitive I/O path, page-cache behavior, explicit correctness bugs, and request-granularity issues, not by advanced cache theory.

## Explicit non-goals
- proving that zero-cache Windows decode is viable on consumer NVMe
- polishing report prose before evidence exists
- evaluating DirectStorage before the cached baseline is understood
- evaluating predictor schemes before finite lead time and recall-vs-budget are measured
- assuming that KV compression alone solves expert streaming

## Deliverables
1. Systems-baseline package (correctness harness, I/O primitive bakeoff, first affine timing fits)
2. Routing trace package
3. Timing calibration package
4. Simulator outputs across policy families
5. Critical cache size analysis
6. KV-compression scenario analysis
7. Layerwise `K_l` tradeoff study
8. Recommendation on prediction and DirectStorage headroom
9. Final V2 report

## Success conditions
- evidence-backed go / conditional-go / no-go judgment
- clear description of what makes the design viable or non-viable
- explicit list of assumptions that remain unmeasured, if any
