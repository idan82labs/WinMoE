# Timing Analysis 001 — First SSD + PCIe Measurements

Date: 2026-03-25
Drive: Samsung SSD 980 1TB (D:)
GPU: RTX 3070 Laptop, 8 GB VRAM

## SSD Read Findings (Samsung 980 NVMe)

### Sequential reads — explicit unbuffered (FILE_FLAG_NO_BUFFERING)
| Request size | Bandwidth (MB/s) | Per-request (µs) |
|-------------|-------------------|-------------------|
| 4 KB        | 90                | 44                |
| 16 KB       | 322               | 49                |
| 64 KB       | 956               | 65                |
| 256 KB      | 1,070             | 234               |
| 1 MB        | 1,722             | 581               |
| 4 MB        | 2,500             | 1,600             |
| 16 MB       | 2,670             | 5,993             |
| 64 MB       | 2,853             | 22,430            |

### Sequential reads — explicit buffered (OS cache)
Peak: ~6,774 MB/s at 256 KB (reading from OS page cache, not SSD)

### Sequential reads — mmap
Consistent ~1,000–1,100 MB/s for all sizes. This is misleading because test file fits in RAM page cache. For real MoE expert files >> RAM size, mmap will degrade to page-fault-dominated performance.

### Random reads — unbuffered
| Request size | Bandwidth (MB/s) | Per-request (µs) |
|-------------|-------------------|-------------------|
| 4 KB        | 45                | 87                |
| 64 KB       | 469               | 133               |
| 256 KB      | 994               | 252               |
| 1 MB        | 1,815             | 551               |
| 4 MB        | 2,457             | 1,628             |

### Affine fit — SSD unbuffered
- alpha ≈ 0 µs (startup absorbed into beta)
- **beta ≈ 42 µs per request**
- **B ≈ 2,118 MB/s = 2.07 GB/s asymptotic bandwidth**

## PCIe Transfer Findings (RAM→GPU)

### Pinned memory H2D
| Transfer size | Bandwidth (MB/s) | Per-transfer (µs) |
|--------------|-------------------|--------------------|
| 4 KB         | 328               | 12                 |
| 16 KB        | 1,454             | 11                 |
| 64 KB        | 5,645             | 11                 |
| 256 KB       | 16,331            | 15                 |
| 1 MB         | 15,161            | 66                 |
| 4 MB         | 18,599            | 215                |
| 16 MB        | 18,363            | 871                |
| 64 MB        | 19,221            | 3,330              |
| 128 MB       | 16,824            | 7,608              |

### Pageable memory H2D
Peak ~8,300 MB/s — roughly 2.3× slower than pinned.

### Key PCIe parameters
- **Pinned asymptotic bandwidth: ~18–19 GB/s**
- **Per-transfer overhead (pinned): ~11 µs**
- **Pageable bandwidth: ~6–8 GB/s**

## Implications for MoE Expert Streaming

### Per-layer miss scenario (Qwen3-235B style, K=10, ~6.75 MB/expert)
Full miss payload: D = K × 6.75 = **67.5 MB/layer**

| Stage | Time at full miss |
|-------|-------------------|
| SSD read (67.5 MB, unbuf) | ~24–32 ms |
| PCIe transfer (67.5 MB, pinned) | ~3.5 ms |
| **Total miss-service (causal)** | **~28–36 ms** |

### Bottleneck analysis
- SSD is clearly the bottleneck (6–9× slower than PCIe)
- At 28-36 ms per layer, a 94-layer model gives ~2.6–3.8 seconds per token → **0.3–0.4 tok/s** without caching
- This confirms the spec's conclusion: **zero-cache Windows decode is not viable**
- RAM-first caching that keeps SSD miss rate < 5% is necessary for usable tok/s

### Request size sensitivity
- **Critical finding**: SSD unbuffered bandwidth drops from 2.9 GB/s (64 MB) to 90 MB/s (4 KB)
- Expert blocks should be packed as contiguous slabs, not scattered extents
- Minimum practical request size for reasonable bandwidth: **≥256 KB**
