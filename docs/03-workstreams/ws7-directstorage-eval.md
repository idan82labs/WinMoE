# WS7 — DirectStorage Evaluation

## Objective
Decide whether DirectStorage has enough remaining headroom to justify the integration effort.

## Required inputs
- cached baseline already measured
- residual SSD/control-path fraction of layer time
- request-count regime after slabification and caching

## Required outputs
- Amdahl-style upside estimate
- decision memo: later optimization vs unnecessary complexity

## Exit criteria
DirectStorage has a clear project rank: worthwhile later-stage optimization or not worth it on the current design.
