# Evidence Required Per Report Section

## Systems-baseline bring-up section
- correctness checks against reference path
- primitive I/O comparison notes
- cached vs uncached observations where tested
- first affine timing fits

## Trace / cacheability section
- per-layer marginals
- top-set stability
- working-set growth
- reuse-distance summaries

## Policy / tier-sizing section
- static vs recency vs hybrid vs oracle results
- service-demand curves and critical cache sizes

## Memory-budget / `K_l` section
- KV scenario tables
- retained-mass or stronger surrogate curves
- scenario comparisons

## Prediction / DirectStorage section
- finite lead-time analysis
- recall-vs-budget curves if prediction is tested
- Amdahl-style residual-headroom estimate for DirectStorage
