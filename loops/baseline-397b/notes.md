# Notes

## Trial 1 — reproducibility
Rerun matches baseline exactly: 1.22 tok/s, 821 ms/tok, TTFT 7504ms (vs 7519ms baseline). Variance <1%. Baseline is stable.

## Trial 2 — K=6
1.15 tok/s (-5.7%). Costs 45 ms/token extra. Without measured quality improvement to justify the regression, reject. Lesson: don't increase K without quality evidence. K=5 baseline retained.

## Trial 3 — CUDA llama.cpp build
**BLOCKED.** Missing: CUDA toolkit (nvcc), MSVC compiler (cl.exe). No pre-built CUDA wheel exists for Python 3.13 + Windows. The CUDA index at abetlen.github.io serves the same CPU-only wheel from PyPI. To unblock: install CUDA Toolkit + Visual Studio Build Tools with C++ workload. This is the single biggest remaining optimization lever.
