# Baseline - Timing Fit Loop

## Current baseline — First affine fits (2026-03-25)

### SSD unbuffered read (Samsung 980 NVMe PCIe 3.0)
- alpha ≈ 0 µs (absorbed into beta)
- beta ≈ 42 µs per request
- B ≈ 2,118 MB/s (2.07 GB/s) asymptotic bandwidth
- Fit from: 6 data points (request-count sweep, 64 KB–64 MB)

### PCIe H2D transfer (pinned memory, RTX 3070 Laptop)
- Per-transfer overhead: ~11 µs
- Asymptotic bandwidth: ~18–19 GB/s (pinned)
- Pageable bandwidth: ~6–8 GB/s
- Fit source: size sweep + count sweep

### Implications
- SSD is the bottleneck by 6–9× vs PCIe
- Minimum efficient SSD request: ≥256 KB for >1 GB/s
- Pinned memory is mandatory for async H2D
