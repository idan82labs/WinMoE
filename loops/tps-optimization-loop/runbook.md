# Runbook - TPS Optimization Loop

## Benchmark command
```
python work/baseline/run_397b.py \
  --model-path <gguf_path> \
  --n-gpu-layers <N> \
  --ctx 512 \
  --max-tokens 50 \
  --prompt "Explain the theory of general relativity and its implications for modern physics."
```

## Loop steps
1. Record baseline tok/s from last accepted run
2. Make ONE change from the optimization queue:
   - Change K value (patch_k_experts.py)
   - Change n_gpu_layers
   - Change context length
   - Change quantization level
   - Change thread count
   - Enable/disable specific llama.cpp flags
3. Run benchmark command
4. Record: change, tok/s, quality check, decision
5. If tok/s improves without quality regression: KEEP, update baseline
6. If tok/s regresses or quality degrades: REJECT, revert

## Optimization queue (priority order)
1. [x] K=10 baseline: 0.44 tok/s
2. [x] K=7: 0.82 tok/s (KEEP)
3. [x] K=5: 1.00 tok/s (KEEP)
4. [ ] GPU offload: --n-gpu-layers 5
5. [ ] Thread count optimization: --threads <N>
6. [ ] Context reduction: --ctx 256
7. [ ] Layerwise K_l schedule (requires llama.cpp modification)
8. [ ] CUDA build of llama.cpp for GPU compute
9. [ ] Better quant: upgrade from IQ2_XXS to Q4_K_S
10. [ ] Batch prefetch: overlap SSD reads with compute
