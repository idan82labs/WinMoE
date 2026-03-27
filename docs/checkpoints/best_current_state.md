# Frozen Best Speed Baseline — current_best_speed_baseline

Date: 2026-03-26

## Exact Configuration

- **Model**: Qwen3.5-397B-A17B
- **Quantization**: UD-IQ2_XXS (Unsloth Dynamic ~2-bit, importance-matrix)
- **GGUF path**: `D:/hf_cache/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_XXS/Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf`
- **K (active experts)**: 3 (patched via patch_k_experts.py)
- **n_gpu_layers**: 8
- **threads**: 12
- **ctx_size**: 512
- **Engine**: D:/llama-cpp-cuda/bin/llama-cli.exe (CUDA build, RTX 3070)
- **Cache state**: warm (model files resident in OS page cache)

## Exact Command

```bash
D:/llama-cpp-cuda/bin/llama-cli.exe \
  --model "D:/hf_cache/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_XXS/Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf" \
  --n-gpu-layers 8 --ctx-size 512 --n-predict 200 --threads 12 \
  --prompt "What is quantum computing?" \
  --no-warmup --no-display-prompt --single-turn --reasoning off
```

## Metrics

- **TTFT**: ~450 ms (prompt eval 2.2 tok/s on 1 token)
- **Generation tok/s**: 1.6 (50 tokens), 1.7 (200 tokens), 1.7 (500 tokens)
- **Sustained tok/s (200+ tokens)**: **1.7**
- **Prompt eval tok/s**: 2.0-2.2
- **ms per token (decode)**: 588-625

## Sustained Benchmark Details

| Token count | Gen tok/s | Prompt tok/s |
|-------------|-----------|-------------|
| 50 | 1.5-1.6 | 2.2 |
| 200 | 1.6-1.7 | 2.0 |
| 500 | 1.7 | 1.2 |

Throughput improves with generation length due to OS page cache warming.

## Output Sample (K=3, 200 tokens)

> **Quantum computing** is an emerging field of computing that uses the principles of quantum mechanics to solve complex problems that are too difficult for classical computers to handle efficiently. Unlike classical computers, which use bits (0 or 1) as the smallest unit of data, quantum computers use **quantum bits**, or **qubits**. [...] a qubit can be 0, 1, or **both at the same time**. This allows quantum computers to process a vast number of possibilities simultaneously rather than sequentially.

Coherent, well-structured, factually correct, uses markdown formatting.

## Hardware

- CPU: Intel laptop (16 logical cores)
- GPU: NVIDIA GeForce RTX 3070 Laptop GPU, 8 GB VRAM
- RAM: 39.7 GB DDR4
- SSD: Samsung 980 1TB NVMe PCIe 3.0 (~2.1 GB/s)

## Reproducibility

1. Ensure GGUF is patched to K=3: `python work/baseline/patch_k_experts.py <gguf_path> --k 3 --apply`
2. Warm the page cache: run inference once, discard result
3. Run the exact command above
4. Expected: 1.6-1.7 tok/s generation

## Improvement History

| Stage | tok/s | Multiplier |
|-------|-------|-----------|
| Original (K=10, CPU, cold) | 0.44 | 1.0x |
| K=5, CPU, cold | 1.00 | 2.3x |
| K=5, CPU, warm | 1.22 | 2.8x |
| K=5, ngl=8, warm | 1.3 | 3.0x |
| K=3, ngl=8, t=12, warm | **1.7** | **3.9x** |
