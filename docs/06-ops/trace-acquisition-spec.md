# Trace Acquisition Spec

## Goal
Define a faithful path to collect per-token, per-layer expert-activation data suitable for simulation.

## Must be decided explicitly
- target model
- target runtime / framework
- instrumentation location in the forward path
- logging overhead tolerance
- workloads to trace
- storage format

## Minimum fields per event or record
- workload / prompt identifier
- token index
- layer index
- active expert ids
- optional gate weights if accessible
- timestamp or ordering index

## Derived summaries needed downstream
- `q_{l,e}(t)` or windowed marginals
- top-set stability at VRAM and RAM cutoffs
- working-set growth
- consecutive-token overlap
- reuse-distance or equivalent recency structure

## Blocking questions to answer early
- can traces be collected on the target runtime without unacceptable distortion?
- if not, is a proxy runtime faithful enough for policy work?
- does the chosen trace path align with the primitive I/O baseline selected in WS0?
