# Phase Plan

## Phase -1 — Workspace and operating model
Exit when:
- repo scaffold is in place
- workstreams, gates, templates, and state files exist
- the team agrees on execution order

## Phase 0 — Systems baseline
Exit when:
- a correctness-first reference path exists or the blocker is crisply documented
- explicit file-I/O primitive bakeoff results exist
- first slab-layout and cache/no-cache measurements exist
- first affine timing fits exist for the primitive path
- the project has chosen its default expert-streaming primitive stack

## Phase 1 — Observability and calibration
Exit when:
- routing trace path is defined and tested
- timing harness is defined and expanded from the phase-0 baseline
- target-faithful trace collection is judged feasible or explicitly blocked

## Phase 2 — Simulator V0
Exit when:
- simulator accepts trace + timing inputs
- static, recency, hybrid, and oracle-like comparisons can run

## Phase 3 — Feasibility and tier sizing
Exit when:
- service-demand curves are produced
- critical cache sizes are estimated
- safe / unsafe hardware regions are identified

## Phase 4 — Memory-budget and `K_l` levers
Exit when:
- KV-compression scenarios are modeled into tier budgets
- `K_l` candidates are studied
- the combined value of memory-budget shrinkage vs active-expert reduction is understood

## Phase 5 — Prediction and DirectStorage headroom
Exit when:
- prediction is judged by finite lead time and recall-vs-budget
- DirectStorage headroom is estimated with Amdahl-style logic

## Phase 6 — V2 lock
Exit when:
- final recommendation is supported by evidence
- open assumptions are explicitly marked
- go / conditional-go / no-go language is ready


## Added Phase 0 requirement - inner loops

Phase 0 now includes autoresearch-style inner loops. Before leaving the systems-baseline phase, the I/O path loop and timing fit loop should exist, and the I/O path loop should have at least one recorded baseline versus candidate comparison. Gate 0B captures this readiness check.
