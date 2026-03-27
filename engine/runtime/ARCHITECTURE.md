# WinMoE Custom Engine Architecture

## Decision: Pure C with AVX512 intrinsics

No GGML, no llama.cpp, no frameworks. Pure C for maximum control over:
- Memory allocation (no mmap, explicit buffers)
- I/O path (unbuffered Windows API)
- Compute (AVX512 intrinsics for matmul + dequant)
- Pipeline scheduling (manual threading)

## IQ2_XXS Dequantization — Solved

Studied the llama.cpp source. IQ2_XXS is simple:
- 256-weight super-blocks with FP16 scale `d`
- Each 32-weight sub-block has: 4 grid indices (8 bits each) + sign bits + sub-scale
- Grid LUT: `iq2xxs_grid[256]` — each entry is 8 uint8 values
- Sign LUT: `ksigns_iq2xs[128]` — encodes 7 sign bits for 8 values
- Dequant: `value = scale * sub_scale * grid_value * sign`

This is pure integer lookup + multiply. Fast on AVX512.

## Compute Operations Needed

For one token through 60 layers:

### Per layer:
1. **RMSNorm**: `x = x * w / sqrt(mean(x^2) + eps)` — trivial
2. **Attention** (simplified for v0):
   - Q = x @ W_q, K = x @ W_k, V = x @ W_v (IQ2_XXS matmul)
   - RoPE on Q and K
   - Score = Q @ K^T / sqrt(d), softmax, out = score @ V
   - O = out @ W_o
3. **Post-attention norm**: RMSNorm
4. **Router**: x @ W_gate → softmax → topK=3
5. **Expert FFN** (×3 experts):
   - gate = x @ W_gate_expert (IQ2_XXS)
   - up = x @ W_up_expert (IQ2_XXS)
   - out = SwiGLU(gate, up) @ W_down_expert (IQ2_XXS)
6. **Expert combine**: weighted sum of 3 expert outputs + residual

### Final:
7. **Final norm**: RMSNorm
8. **LM head**: hidden @ W_vocab → logits → sample

## Core Kernel: IQ2_XXS MatVec

The critical operation. For a (M, N) weight matrix in IQ2_XXS and input vector x[N]:

```c
// For each output row m:
//   y[m] = sum over n: dequant(W[m,n]) * x[n]
//
// With IQ2_XXS packing:
//   Block of 256 weights, process 32 at a time
//   Each 32-weight chunk: 4 grid lookups × 8 values each
```

AVX512 can process 16 floats at once. Each grid lookup gives 8 values.
Two grid lookups = 16 values = one AVX512 register.

## Build Phases

### Phase 0: Skeleton (current)
- [x] Slab reader with explicit I/O
- [x] Tiered RAM cache
- [x] Streaming scheduler simulation
- [ ] IQ2_XXS dequant function (port from llama.cpp, ~40 lines)
- [ ] Basic matmul: dequant + dot product

### Phase 1: Single expert forward pass
- [ ] Load one expert's gate/up/down weights from slab
- [ ] Dequant to FP32
- [ ] MatVec: gate = W_gate @ x, up = W_up @ x
- [ ] SwiGLU: out = silu(gate) * up
- [ ] MatVec: result = W_down @ out
- [ ] Benchmark: ms per expert FFN

### Phase 2: Full MoE layer
- [ ] Router forward (small linear projection)
- [ ] TopK expert selection
- [ ] Load K=3 experts from slab/cache
- [ ] Compute all 3 expert FFNs
- [ ] Weighted combine + residual
- [ ] Benchmark: ms per layer

### Phase 3: Attention (simplified)
- [ ] Load attention weights from GGUF (shared, always resident)
- [ ] Q/K/V projections
- [ ] RoPE
- [ ] Dot-product attention (no flash attention needed for batch=1)
- [ ] Output projection

### Phase 4: Full forward pass
- [ ] Chain 60 layers
- [ ] LM head
- [ ] Sampling (greedy or temperature)
- [ ] Token output
- [ ] Benchmark: actual tok/s

### Phase 5: Autoresearch optimization
- [ ] AVX512 optimized matmul
- [ ] Fused dequant+matmul kernel
- [ ] Within-layer expert streaming (async I/O overlap)
- [ ] Cache warming strategies
- [ ] Thread pool for parallel expert compute

## File Structure
```
engine/runtime/
  engine.c          — main inference loop (autoresearch target)
  iq2_dequant.h     — IQ2_XXS dequantization (LUTs + dequant function)
  matvec.h          — matrix-vector multiply with quantized weights
  attention.h       — attention mechanism
  moe.h             — MoE routing + expert FFN
  slab_io.h         — explicit I/O from slab file
  cache.h           — tiered expert cache
  tokenizer.h       — BPE tokenizer (or shell out)
  benchmark.py      — immutable scoring harness
  program.md        — autoresearch instructions
  results.tsv       — experiment ledger
```
