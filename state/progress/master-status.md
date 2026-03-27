# Master Status — WinMoE Custom Engine

## Current Performance
**2.60 tok/s** on Qwen3.5-397B Q4_K_M (228 GB, 6 GGUF shards)
**52x speedup** from first token (0.05 tok/s at v5.0)

## Architecture
```
SSD (228 GB Q4_K_M) → RAM (20 GB expert cache, ~80% hit) → CPU (AVX-512+VNNI expert FFN)
                                                           → GPU (Q8_0 DeltaNet projections, 6.1 GB VRAM)
```

## Build Command
```bash
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin:$PATH"
cd engine/runtime

# In powershell with vcvars:
vcvarsall.bat x64
nvcc -O2 -arch=sm_86 -c gpu_offload.cu -o gpu_offload.obj
cl /O2 /arch:AVX512 /openmp /fp:fast /c winmoe_inference.c /Fo:winmoe.obj
link winmoe.obj gpu_offload.obj /OUT:winmoe.exe /STACK:8388608 cublas.lib cuda.lib cudart.lib /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64"

# Run (must have CUDA bin in PATH):
OMP_NUM_THREADS=10 ./winmoe.exe --model "D:/models/qwen35-397b-q4/Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-00001-of-00006.gguf" --tokens 10
```

## Profiled Breakdown (best token, 384ms)
- DeltaNet attention: 176ms (46%) — GPU custom Q8_0 kernel
- Expert FFN: 182ms (47%) — CPU AVX-512 integer accumulation, K=4
- Router: 24ms (6%)

## Version History
| Version | tok/s | Speedup | Change |
|---------|-------|---------|--------|
| v5.0 | 0.05 | 1x | First 397B token |
| v5.7 | 0.67 | 13x | Buffer bug fix |
| v6.3 | 0.91 | 18x | AVX-512 |
| v6.5 | 1.39 | 28x | Integer accumulation + fusion |
| v7.0 | 1.53 | 31x | VNNI |
| v7.4 | 2.60 | 52x | GPU Q8_0 custom CUDA kernel |

## Phase 3 Finding
ML expert determined GPU-CPU pipeline gives only 3-5% (serial dependency chain).
Real path to 4.8+ tok/s: VNNI expert kernels + GPU expert caching + async prefetch.

## Next Actions
1. Implement Phase 3 async infrastructure (from PHASE3_PIPELINE_PLAN.md)
2. Phase 4: VNNI for Q4_K/Q5_K expert kernels
3. GPU expert caching (1.9 GB spare VRAM = ~250 hot experts)
4. GQA attention for standard layers (15 layers output zeros)
5. LM head on GPU for real text output

## Key Files
- engine/runtime/winmoe_inference.c — main loop
- engine/runtime/gpu_offload.cu — CUDA kernels
- engine/runtime/results.tsv — 42 experiments
- engine/runtime/PHASE3_PIPELINE_PLAN.md — pipeline design
- docs/paper_draft.md — research paper draft
- STUDY.md — full research narrative
