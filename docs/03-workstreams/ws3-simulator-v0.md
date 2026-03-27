# WS3 — Simulator V0

## Objective
Build the smallest useful simulator that maps traces + timing fits to policy outcomes.

## Required policy set
- static hotset
- recency buffer
- hybrid static + adaptive
- oracle reference if feasible

## Required outputs
- SSD bytes and requests
- RAM-to-GPU bytes and copies
- estimated service-demand distribution
- predicted safe / unsafe cache regions

## Exit criteria
The simulator can run the same workload under multiple policies and produce comparable outputs.
