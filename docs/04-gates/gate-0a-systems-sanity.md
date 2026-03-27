# Gate 0A — Systems Sanity

Pass only if:
- a correctness-first reference path exists or the blocker is crisply documented
- explicit I/O primitive comparisons exist (`mmap`/paging-style vs explicit reads where relevant)
- cache/no-cache or equivalent OS-path behavior has been tested at least once
- slab-layout / request-granularity observations exist
- first affine timing fits exist for the chosen baseline path
- the selected default primitive stack is written down in state and docs

This gate exists to prevent the project from building a simulator on top of the wrong primitive stack.
