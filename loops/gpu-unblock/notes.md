# GPU Unblock — Notes

## 2026-03-26 ~02:10 — CUDA GPU sweep complete

Used pre-built CUDA binary at D:/llama-cpp-cuda/bin/llama-cli.exe.
RTX 3070 Laptop 8GB VRAM detected and working.

### GPU layer sweep results (K=5, warm cache):

| n_gpu_layers | Generation tok/s | Prompt tok/s | vs CPU baseline |
|-------------|-----------------|-------------|-----------------|
| 0 (CPU) | 1.22 | 1.22 | baseline |
| 5 | 1.0 | 1.3 | -18% |
| 6 | 1.1 | 1.4 | -10% |
| 7 | 1.2 | 1.4 | -1.6% |
| **8** | **1.3** | **1.2** | **+6.6%** |
| 9 | 1.1 | 1.2 | -10% |
| 10 | 1.2 | 1.3 | -1.6% |
| 11 | 1.1 | 1.3 | -10% |
| 12 | 1.0 | 1.3 | -18% |
| 15 | 1.0 | 1.3 | -18% |

### Key findings:
1. **ngl=8 is the sweet spot**: 1.3 tok/s generation, +6.6% over CPU-only
2. **The improvement is modest** — SSD expert streaming is the bottleneck, not compute
3. **ngl>8 degrades**: VRAM overflow spills back to CPU with added overhead
4. **A faster NVMe remains the #1 hardware lever**
