# Systems Baseline Spec

## Goal
Choose the primitive stack that later phases will treat as the default baseline.

## Required comparisons
- explicit reads vs mapped/paging-style access if both are available
- cached vs uncached expert-file path where supported
- contiguous slab layout vs fragmented extents
- representative request-size sweeps
- correctness checks for shared-expert handling and cache invalidation logic

## Minimum outputs
- chosen baseline path
- rejected alternatives and why they were rejected
- early timing fits for the chosen path
- correctness status against a trusted reference path

## Decision rule
Do not advance to broad simulator claims until a baseline path has been selected or the inability to select one is clearly documented.
