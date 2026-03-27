# WS5 — K Reduction and Layerwise `K_l`

## Objective
Study whether reducing active experts, especially layerwise, yields better feasibility gains per byte than more complex systems work.

## Questions answered
- what do `p_l(K)` and stronger surrogates imply for byte savings vs quality risk?
- does layerwise `K_l` buy more than prediction or DirectStorage in the near term?
- what combinations of KV compression + `K_l` alter the feasible cache envelope?

## Inputs
- simulator-derived cache sensitivity
- baseline / compressed-KV memory envelopes
- retained-mass or retained-energy statistics

## Required outputs
- retained-mass curves
- stronger surrogate curves if available
- candidate layerwise `K_l` schedules
- scenario comparison notes

## Exit criteria
The project can rank `K_l` against other second-order levers using evidence.
