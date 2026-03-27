# GPU Unblock — Frozen CPU Baseline

## Command
```
python work/baseline/run_397b.py \
  --model-path "D:/hf_cache/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_XXS/Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf" \
  --n-gpu-layers 0 --ctx 512 --max-tokens 50 \
  --prompt "What is quantum computing?"
```

## Model
- Path: D:/hf_cache/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_XXS/Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf
- Quant: UD-IQ2_XXS
- K: 5 (patched)

## Cache state
- Warm (model files in OS page cache from prior runs)

## Runtime / build
- llama-cpp-python 0.3.18, CPU-only wheel (no CUDA)
- Python 3.13.1 Windows 11
- GPU offload: NOT active (CPU build ignores n_gpu_layers)

## Hardware
- CPU: 16 logical cores (laptop)
- GPU: NVIDIA GeForce RTX 3070 Laptop, 8 GB VRAM, driver 580.88, CUDA 13.0
- RAM: 39.7 GB DDR4
- SSD: Samsung 980 1TB NVMe PCIe 3.0

## Metrics (verified reproducible)
- TTFT: 7,504 ms
- tok/s: 1.22
- ms/token: 821
