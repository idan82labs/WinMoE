# WS1 — Trace Acquisition

## Objective
Capture routing traces that are faithful enough to drive policy simulation.

## Precondition
WS0 systems baseline has selected a default runtime / primitive path, or any deviation is explicitly justified.

## Questions answered
- what is the per-layer expert popularity profile?
- how stable are the top sets at the RAM and VRAM cutoffs?
- how quickly do working sets grow?
- how much short-lag reuse exists?

## Inputs
- target model/runtime choice
- instrumentation access point
- workload set

## Required outputs
- logging schema
- instrumentation plan or implementation notes
- top-set stability metrics
- working-set growth curves
- reuse-distance summaries
- evidence registration

## Exit criteria
A trace package exists that is detailed enough for the simulator to consume or the blocking reason is crisply documented.
