# WS0 — Systems Baseline

## Objective
Establish a correctness-first, explicit-I/O baseline before broad trace-driven policy work begins.

## Why it exists
Recent external local-MoE bring-up evidence suggests the first large wins often come from fixing the primitive storage path, page-cache behavior, slab layout, and correctness issues rather than from advanced caching or prediction.

## Questions answered
- what is the default expert-streaming primitive stack?
- is `mmap` or paging-style access obviously inferior to explicit reads?
- how much do cache/no-cache modes, slab layout, and request granularity matter?
- are correctness bugs contaminating performance conclusions?

## Required outputs
- correctness regression harness notes
- primitive I/O bakeoff results
- page-cache / write-back observations
- slab-layout experiment notes
- first affine timing fits for the chosen primitive path
- recommended baseline path for later phases

## Exit criteria
A default baseline path is selected and documented, or the reason a clean baseline cannot be established is crisply documented.
