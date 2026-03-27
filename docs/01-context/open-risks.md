# Open Risks

## Technical risks
- routing traces may be hard to obtain cleanly from a realistic MoE runtime
- top-set stability may be weaker than hoped
- affine overhead terms may dominate at practical transfer sizes
- simulator results may be highly assumption-sensitive
- richer `K_l` surrogates may be too costly to compute in practice
- primitive I/O choices may dominate enough that early simulator assumptions become stale
- local bring-up may fail first on correctness bugs, shared-expert handling, or cache invalidation issues

## Process risks
- spending too much time refining documents instead of collecting evidence
- mistaking simulator design for simulator outputs
- letting prediction or DirectStorage work start before the cached baseline exists
- over-reading external success stories without separating hardware, model, and runtime differences

## Mitigation
Use validation gates and explicit evidence requirements.
