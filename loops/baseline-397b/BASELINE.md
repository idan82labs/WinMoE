# Frozen Baseline

## Command
```
python work/baseline/run_397b.py \
  --model-path "D:/hf_cache/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_XXS/Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf" \
  --n-gpu-layers 0 --ctx 512 --max-tokens 50 \
  --prompt "What is quantum computing?"
```

## Model
- Path: `D:/hf_cache/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_XXS/Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf`
- Quant: UD-IQ2_XXS (Unsloth Dynamic ~2-bit, importance-matrix)
- K: 5 (patched via patch_k_experts.py)

## Cache state
- Warm: model files resident in OS page cache from prior runs
- To reproduce warm state: run the model once, then measure on the second run

## Runtime / build
- Engine: llama-cpp-python 0.3.18
- Backend: **CPU only** (no CUDA compiled in)
- Python: 3.13.1, Windows 11
- GPU offload: **NOT active** (n_gpu_layers accepted but ignored by CPU build)

## Hardware
- RAM: 39.7 GB DDR4
- GPU: RTX 3070 Laptop 8 GB (not used)
- SSD: Samsung 980 1TB NVMe PCIe 3.0 (~2.1 GB/s measured)

## Metrics
- TTFT: 7,519 ms
- tok/s: 1.24 (warm, decode)
- ms/token: 806
