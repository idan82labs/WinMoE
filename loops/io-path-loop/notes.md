# Notes - I/O Path Loop

## 2026-03-25 — Initial bakeoff

Three methods compared on Samsung 980 1TB NVMe (PCIe 3.0):

1. **Explicit unbuffered (FILE_FLAG_NO_BUFFERING)** → SELECTED as baseline
   - Gives true SSD bandwidth (2.1–2.9 GB/s depending on request size)
   - Per-request overhead ~42 µs (beta term)
   - Full control over what/when/where to read
   - Async capable via OVERLAPPED I/O

2. **Explicit buffered** → REJECTED for cold-path baseline
   - 6.8 GB/s peak is OS page cache speed, not SSD
   - Misleading for expert files that don't fit in RAM
   - May be useful later as a cached-tier path metric

3. **mmap** → REJECTED
   - ~1 GB/s with test file in page cache
   - Will thrash catastrophically for >>RAM workloads (MoE experts = 100-189 GB)
   - No async control; no prefetch control; 4KB fault granularity is wasteful
   - External evidence confirms mmap is a poor choice for MoE expert streaming

## Key lesson
Request size is the dominant factor for SSD bandwidth. At 4KB: 90 MB/s. At 64MB: 2.9 GB/s. Expert blocks must be stored as contiguous slabs (minimum ~256KB to reach >1 GB/s).
