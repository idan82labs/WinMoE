# Notes - TPS Optimization Loop

## 2026-03-25 — Initial K sweep
- K=10→5 gives clean 2.3x speedup (0.44→1.00 tok/s)
- Scaling is nearly linear with K: each expert costs ~228 ms (1000ms/5 - 2281ms/10... roughly)
- Compute is not the bottleneck — expert weight loading from SSD dominates

## Next experiments to try (from V2 spec optimization levers)

### Quick wins (no code changes needed)
1. GPU offload (--n-gpu-layers): put attention layers on GPU, keep experts on CPU/SSD
2. Thread count: llama.cpp defaults may not be optimal for this CPU
3. Context reduction: shorter ctx means less KV cache memory, more room for expert cache

### Medium effort (config/patch changes)
4. Layerwise K_l: vary K per layer (needs either llama.cpp modification or per-layer GGUF patching)
5. Expert prefetch: llama.cpp may have internal prefetch settings

### High effort (requires building from source)
6. CUDA build of llama.cpp: enables GPU offload for attention + overlap with SSD I/O
7. Custom expert streaming engine: full control over I/O pipeline
