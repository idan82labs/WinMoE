# Timing Calibration Spec

## Stage model
For each relevant stage, fit:

`t(x, m) = alpha + beta * m + x / B`

where:
- `x` = bytes transferred or processed
- `m` = number of requests, copies, kernels, or extents
- `alpha` = fixed startup/synchronization cost
- `beta` = per-request overhead
- `B` = asymptotic bandwidth

## Required calibration targets
- SSD contiguous reads
- SSD fragmented or multi-extent reads
- cached vs uncached file modes where supported
- RAM-to-GPU transfers at realistic sizes/counts
- dequant or unpack timing
- synchronization/fence overhead
- any page-fault or mapped-path timing if such a path is tested

## Useful derived quantity
Minimum efficient transfer size:
`x* ~= alpha * B`

## Why this matters
This prevents false confidence from headline bandwidth numbers when real transfer sizes are small or request counts are large.
