# Agent Team Topology

## Core orchestration
- `v2-orchestrator` — coordinates phases, sequencing, and gates

## Systems-baseline team
- `io-path-sanity-agent` — primitive I/O bakeoff, cache/no-cache behavior, slab layout
- `skeptical-reviewer` — correctness and hidden-assumption pressure testing

## Measurement team
- `routing-trace-analyst` — routing observability and trace summaries
- `timing-calibrator` — affine timing fits and calibration notes

## Simulation / decision team
- `cache-simulator` — simulator implementation and scenario runs
- `policy-benchmarker` — static vs recency vs hybrid vs oracle comparisons
- `kv-compression-analyst` — memory-budget scenario changes from compressed KV
- `k-reduction-analyst` — retained-mass / retained-energy and `K_l`
- `predictor-evaluator` — finite lead time and recall-vs-budget analysis

## Synthesis team
- `docs-synthesizer` — updates docs/state/report sections from evidence
- `skeptical-reviewer` — final challenge before gate lock


## Added team - Loop Ops

Purpose: run local autoresearch-style bounded loops inside major workstreams.

Members:
- self-improvement-orchestrator
- io-path-sanity-agent
- skeptical-reviewer
- docs-synthesizer
