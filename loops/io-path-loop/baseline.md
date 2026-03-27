# Baseline - I/O Path Loop

## Current baseline

Method: explicit unbuffered reads (FILE_FLAG_NO_BUFFERING via Win32 CreateFileW/ReadFile)
Drive: Samsung SSD 980 1TB (NVMe PCIe 3.0)
Date: 2026-03-25

### Measured performance (sequential, 64 MB request)
- Bandwidth: **2,853 MB/s**
- Per-request overhead: ~22.4 ms per 64 MB request

### Affine fit
- alpha ≈ 0 µs
- beta ≈ 42 µs per request
- B ≈ 2,118 MB/s asymptotic

### Per-layer miss-service estimate (Qwen3.5-397B Q4, D=67.5 MB)
- SSD read: ~24-32 ms
- PCIe transfer (pinned): ~3.5 ms
- Total causal: ~28-36 ms
