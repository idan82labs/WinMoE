# WS2 — Timing Calibration

## Objective
Fit affine stage models for the chosen primitive path and for candidate variants worth testing.

## Questions answered
- what are the startup, per-request, and bandwidth terms for SSD, RAM→GPU, and dequant stages?
- what is the minimum efficient transfer size for each stage?
- how sensitive is the path to fragmentation, extent count, and caching mode?

## Inputs
- WS0 baseline path
- timing harness definitions
- representative request-size and extent-count ranges

## Required outputs
- fitted `alpha`, `beta`, and `B` terms per stage
- cached vs uncached comparisons where relevant
- contiguous vs fragmented comparisons where relevant
- calibration notes and limitations

## Exit criteria
Timing fits exist for the selected baseline path and at least the main plausible alternatives.
