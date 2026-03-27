# WS6 — Prediction Evaluation

## Objective
Judge prediction by finite lead time and recall-vs-prefetch-budget, not by abstract next-token accuracy.

## Required outputs
- usable lead time estimates
- recall-vs-budget curve `R(u)`
- residual miss bytes after prefetch at practical budgets
- expected layer-time reduction under partial overlap

## Exit criteria
Prediction is either justified as a meaningful next lever or clearly deprioritized.
