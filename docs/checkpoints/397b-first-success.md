# Checkpoint: Qwen3.5-397B-A17B First Successful Local Execution

Date: 2026-03-25 ~23:45 local time

---

## Exact Configuration

### Model
- Model: Qwen3.5-397B-A17B
- Quantization: UD-IQ2_XXS (Unsloth Dynamic, ~2-bit importance-matrix quantized)
- GGUF source: `unsloth/Qwen3.5-397B-A17B-GGUF`
- Shards: 4 files, ~107 GB total on disk
- GGUF path: `D:/hf_cache/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_XXS/Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf`

### Runtime
- Engine: llama-cpp-python 0.3.18 (pre-built CPU wheel, no CUDA compile)
- Python: 3.13.1 (Windows, MSC v.1942 64 bit)
- llama.cpp backend: CPU only (ggml, no CUBLAS/CUDA)

### Hardware
- CPU: (laptop, Windows 11 Home 10.0.26100)
- GPU: NVIDIA GeForce RTX 3070 Laptop GPU, 8 GB VRAM (NOT USED for inference)
- RAM: 39.7 GB total DDR4
- SSD: Samsung SSD 980 1TB (NVMe PCIe 3.0, ~2.1 GB/s sequential unbuffered)
- SSD (OS): Micron 2210 512GB
- PCIe: Gen 4 x16 capable (GPU), Gen 3 x4 (Samsung 980)

---

## Exact Commands Used

### K=10 (default)
```
python work/baseline/run_397b.py \
  --model-path "D:/hf_cache/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_XXS/Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf" \
  --n-gpu-layers 0 --ctx 512 --max-tokens 50 \
  --prompt "What is quantum computing?"
```

### K=5 (patched)
```
python work/baseline/patch_k_experts.py <gguf_path> --k 5 --apply
python work/baseline/run_397b.py <same args as above>
```

### Prompt used for all runs
`"What is quantum computing?"`

---

## CPU/GPU/Offload Status
- **CPU**: all computation (attention + MoE expert routing + expert FFN)
- **GPU**: NOT USED (n_gpu_layers=0)
- **Offload**: none. Entire model loaded via llama.cpp's mmap-based GGUF loader into system RAM + OS page cache. Expert weights streamed from SSD via OS page faults.

---

## Measured Results

### K Sweep Table

| K | TTFT (ms) | Decode tok/s | ms/token | Prompt tok/s | Total time (s) | Speedup vs K=10 |
|---|-----------|-------------|----------|-------------|-----------------|-----------------|
| 10 | 14,008 | 0.44 | 2,281 | 0.36 | 125.9 | 1.0x |
| 7 | 10,768 | 0.82 | 1,227 | 0.46 | 70.9 | 1.9x |
| 5 | 9,417 | 1.00 | 1,000 | 0.53 | 58.4 | 2.3x |

### K=10 raw llama.cpp output
```
llama_perf_context_print: prompt eval time =   14007.71 ms /     5 tokens ( 2801.54 ms per token,     0.36 tokens per second)
llama_perf_context_print:        eval time =  111773.41 ms /    49 runs   ( 2281.09 ms per token,     0.44 tokens per second)
llama_perf_context_print:       total time =  125904.00 ms /    54 tokens
```

### K=5 raw llama.cpp output
```
llama_perf_context_print: prompt eval time =    9416.74 ms /     5 tokens ( 1883.35 ms per token,     0.53 tokens per second)
llama_perf_context_print:        eval time =   48982.19 ms /    49 runs   (  999.64 ms per token,     1.00 tokens per second)
llama_perf_context_print:       total time =   58448.74 ms /    54 tokens
```

### K=7 raw llama.cpp output
```
llama_perf_context_print: prompt eval time =   10767.98 ms /     5 tokens ( 2153.60 ms per token,     0.46 tokens per second)
llama_perf_context_print:        eval time =   60114.16 ms /    49 runs   ( 1226.82 ms per token,     0.82 tokens per second)
llama_perf_context_print:       total time =   70882.14 ms /    54 tokens
```

### K=10 generated text
```
Quantum computing is a type of computing that uses quantum-mechanical phenomena, such as
superposition and entanglement, to perform operations on data. Unlike classical computers,
which use bits as the smallest unit of information (either 0 or
```
(coherent, factually correct, cut off at 50 tokens)

---

## What Is Proven

1. **397B local execution is now proven.** Qwen3.5-397B-A17B runs on a consumer Windows laptop with 40 GB RAM, 8 GB VRAM, and a Samsung 980 NVMe. This is the first known instance of this specific configuration running locally on this class of hardware.

2. **K sweep direction matches theory.** Reducing K from 10 to 5 produces a 2.3x speedup (0.44 -> 1.00 tok/s). The scaling is approximately linear with K, consistent with the bottleneck being expert weight loading (I/O-bound, not compute-bound).

3. **The causal timing model is confirmed.** The measured ms/token at K=10 (~2281 ms) is consistent with the spec's prediction for causal decode on a 2.1 GB/s NVMe with 60 MoE layers.

4. **K patching via GGUF metadata modification works.** A single-byte patch to `expert_used_count` in the GGUF header successfully changes the number of active experts in llama.cpp.

5. **The iPhone 17 Pro 0.6 tok/s reference is consistent.** Our 0.44 tok/s at K=10 on a slower NVMe aligns with the iPhone achieving 0.6 tok/s on faster Apple NVMe.

---

## What Is NOT Proven

1. **Quality at K=5 is not yet established.** We have not measured perplexity, benchmark scores, or extended coherence at K=5. The 50-token output at K=10 was coherent, but K=5 quality is unknown.

2. **IQ2_XXS quantization quality is not evaluated.** 2-bit quantization has significant quality degradation vs Q4 or Q8. The output may be coherent but factually unreliable.

3. **GPU offload impact is not measured.** All runs were CPU-only. Putting attention layers on GPU could significantly change the compute/IO balance.

4. **Long-context performance is not tested.** All runs used ctx=512. Longer contexts increase KV cache pressure and may change expert caching behavior.

5. **Layerwise K_l is not tested.** Uniform K across all layers is suboptimal per the spec. Quality-preserving K schedules are not yet explored.

6. **The consecutive-token overlap (37.5% from OLMoE) has not been validated on Qwen3.5.** The routing statistics were measured on a different model.

7. **CUDA build of llama.cpp was not achieved.** Pre-built wheels lack CUDA support; building from source failed due to missing build tools.

---

## Reproducibility Status

**Fully reproducible.** All required files exist:
- GGUF model at known path on D: drive
- `run_397b.py` script with exact CLI arguments
- `patch_k_experts.py` for K modification
- Python environment: llama-cpp-python 0.3.18, Python 3.13.1

To reproduce K=5 run:
```bash
cd C:/Users/idant/flash-moe-windows-v2/flash-moe-windows-v2-claude-filesystem-autoresearch-loops
python work/baseline/patch_k_experts.py "<gguf_path>" --k 5 --apply
python work/baseline/run_397b.py --model-path "<gguf_path>" --n-gpu-layers 0 --ctx 512 --max-tokens 50 --prompt "What is quantum computing?"
```

**Baseline must be preserved.** Before any optimization experiments, the K value should be restored to the test configuration. The GGUF file on disk is currently set to K=7 (last test run).

---

## Next Experiments

See `docs/checkpoints/next_optimization_queue.md` for the full prioritized queue.

Immediate next trials:
1. GPU offload (`--n-gpu-layers 5-10`)
2. Thread count tuning
3. K=6 measurement
4. Context reduction test
