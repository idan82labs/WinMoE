# Engine I/O Loop — Notes

## 2026-03-26 — Explicit I/O standalone: 3.6x faster
- mmap: 586 MB/s on 3.7 MB expert blocks
- explicit unbuffered: 2200 MB/s
- Standalone proof: explicit I/O is definitively faster

## 2026-03-26 — Integrated benchmark: 2.34 tok/s projected
- Full token I/O simulation (180 reads/token): explicit = 282 ms vs mmap = 1049 ms
- With compute overlap: projects 2.34 tok/s (compute-bound)

## 2026-03-26 — Page cache pre-staging wrapper: FAILED

**Background warming during inference HURTS performance (-40%).**
The background reader and llama.cpp's mmap compete for the same NVMe queue.

| Test | Gen tok/s | Notes |
|------|-----------|-------|
| No pre-warm | 1.50 | partially warm cache |
| Full pre-warm then inference | 1.40 | cache already warm, no improvement |
| Background warm during inference | **0.90** | **SSD contention kills performance** |

**Lesson**: Cannot overlay external I/O with mmap-based inference on the same SSD.
This is the same finding as Flash-MoE on Apple Silicon — shared storage controller
means reads cannot be parallelized across consumers.

**Conclusion**: The wrapper approach cannot achieve 2.0+ tok/s. The mmap replacement
must happen INSIDE the inference loop. This requires either:
1. Building llama.cpp from source with modified GGUF loader (needs MSVC + nvcc)
2. Building a standalone inference engine using GGML/cuBLAS directly

**The 3.6x I/O speedup is REAL but can only be captured by replacing mmap, not by
warming the cache alongside it.**

## 2026-03-26 — Source build from latest llama.cpp: 1.8 tok/s CPU-only!

Built llama.cpp from source with MSVC (VS Community 2022). The latest source
has better AVX512 optimizations that match the CUDA binary performance:

| Binary | GPU layers | Gen tok/s |
|--------|-----------|-----------|
| Pre-built CUDA (old) | 8 | 1.5 |
| Source-built CPU (new) | 0 | **1.8** |

**Source-built CPU-only is FASTER than pre-built CUDA.** The AVX512 path
in latest llama.cpp is highly optimized for MoE.

## Direct I/O patch attempt: blocked by split GGUF

Added FILE_FLAG_NO_BUFFERING to Windows file path in llama-mmap.cpp.
Failed on split GGUF (4 shards) — sector alignment requirements conflict
with GGUF header parsing and tensor offset validation. Would need deeper
refactoring of the model loader to separate header I/O from data I/O.

## Professor's guidance (received during build)

The professor confirms:
- Compute ceiling is 2.34 tok/s (T_c = 427 ms)
- After replacing mmap, bottleneck shifts to GPU compute
- **Next lever is reducing T_c, not more I/O optimization**
- Gen4 NVMe does NOT help batch-1 after mmap replacement
- Prediction is a dead end (Fano bound kills it)

Current 1.7-1.8 tok/s is 73-77% of theoretical ceiling.
The remaining ~25% gap is compute-bound, not I/O-bound.
