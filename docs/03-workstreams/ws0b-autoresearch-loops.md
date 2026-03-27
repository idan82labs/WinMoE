# WS0B - Autoresearch-Style Inner Loops

## Objective

Add small, measurable improvement loops inside the major Flash-MoE Windows workstreams.

This workstream exists to force each important subsystem to expose:
- a narrow change surface,
- a benchmark path,
- a primary score,
- and an explicit keep or reject rule.

## Included loops

### Loop 1 - I/O Path Loop
Primary metric: effective expert-byte service time.

### Loop 2 - Timing Fit Loop
Primary metric: held-out error of the affine timing model.

### Loop 3 - Simulator Policy Loop
Primary metric: predicted miss-service demand or projected layer time under a fixed trace and timing bundle.

### Loop 4 - Layerwise K Schedule Loop
Primary metric: bytes or miss-service reduction under a bounded quality-surrogate budget.

### Loop 5 - Predictor Value Loop
Primary metric: residual miss-service demand after predictor-guided prefetch under a fixed budget.

## Exit criteria

This workstream is complete when:
1. all priority loops exist,
2. each has a reproducible baseline,
3. each has a first results.tsv,
4. and the first accepted or rejected iteration has been logged.
