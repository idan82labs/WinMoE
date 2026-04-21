# Prompt-Lookup Decoding (PLD) for WinMoE — Comprehensive Plan (DRAFT)

## Context
WinMoE (Qwen3.5-397B-A17B custom inference) has hit ~0.36-0.40 tok/s after Phase 0-4b optimizations:
- AVX2 LM head (606→148 ms)
- GPU expert LRU + cross-stream race fixes
- OMP DN state recurrence
- Expert IO is now the dominant cost: **1300-1640 ms/token** (60 layers × K=10 experts × 50% miss × 7 MB / ~1 GB/s SSD random)

Per-token breakdown after Phase 4b:
| Component | ms |
|---|---|
| attn (GPU) | 600 |
| **expert (CPU + SSD)** | **1640** |
| lm_head (CPU AVX2) | 148 |
| router | 30 |
| dark | ~270 |
| **Total** | **~2700** |

**The bottleneck is now SSD random reads for cache-miss experts.** No further compute optimization will dominate this. PLD is the right next move because it sidesteps the bottleneck entirely on accepted tokens — **N tokens emitted for the cost of 1 forward pass**, IF we can batch-verify N draft tokens.

## What PLD Does
1. **Lookup**: search the recent context (last 256 tokens, including prompt + generated) for an n-gram match of the last `k` tokens (typically k=2-3).
2. **Draft**: take the next `m` tokens (typically m=3-5) after the match as the speculation.
3. **Verify in batch**: run a SINGLE forward pass with the draft tokens batched (positions p+1..p+m). Compare argmax at each position against the draft.
4. **Accept**: take the longest prefix where draft matches argmax. Generate that many tokens this step.
5. **Fallback**: if 0 accepted, normal single-token generation (cost: same as today + tiny n-gram lookup).

Saxena et al. (2023) report 1.5-2x speedup on code generation, 1.2-1.4x on chat. Coding workloads have high token reuse (function names, keywords, common patterns).

## Why This Is Hard For WinMoE Specifically
1. WinMoE's forward pass is **strictly single-token**. Embeddings, attention KV cache writes, router calls, expert FFN, residuals, LM head — all assume `n_tokens == 1`.
2. Multi-token attention requires:
   - Attention for token T uses K cache positions [0..T-1] + own (T)
   - For batch [T, T+1, ..., T+m-1], each position attends to its prior positions PLUS the previously-batched tokens
   - With causal mask, this is GQA-attention's natural batched mode (which llama.cpp implements)
3. Multi-token MoE requires per-position routing, then either:
   - **Naive**: K experts per position × m positions = K*m expert lookups (no IO savings)
   - **Token-grouped**: pool tokens by expert ID — each unique expert is read once, computed for all tokens that routed to it. Big IO win when tokens share experts.
4. KV cache: WinMoE's `kv_cache_append` adds ONE position. Need to support adding `m` positions at once.

## Estimated Speedup
- Code generation (high reuse): n-gram hit rate ~50%, accept rate ~60% → effective `m_avg = 1.5`. So 0.40 tok/s → 0.60 tok/s (+50%). Could hit ~0.8 with token-grouped MoE.
- Chat (low reuse): n-gram hit rate ~10%, accept rate ~40% → `m_avg = 0.04`. Negligible.
- Repetitive output (e.g., long docstrings, table generation): up to 2-3x.

## Implementation Plan — SCOPE A (Minimal, ship-able)
**Goal**: 1.3-1.5x speedup on code workloads. Avoid touching attention or MoE batching.

### A.1 N-gram suffix matcher
**File**: new `engine/runtime/pld.h`
- Maintain `int context_history[MAX_CTX]` of all token IDs (prompt + generated).
- After each generation step, append.
- `int pld_lookup(history, len, k=3, draft_out, max_m)`: scan backwards in history for k-gram match of `history[len-k..len-1]`. Return up to `max_m=4` tokens after the match. O(len*k) per lookup; for len=512 and k=3, ~1500 comparisons — negligible.
- Returns m=0 if no match, else m=1..4 draft tokens.

### A.2 Verify-by-rerun (NO multi-token batch)
**File**: `engine/runtime/winmoe_inference.c` decode loop
- After computing token T's logits, get argmax → `tok_T`.
- If T+1 < num_tokens: call `pld_lookup` with `history + [tok_T]` to get up to m draft tokens.
- For each draft token (sequentially):
  - Run normal forward pass with draft token as input.
  - Get logits, take argmax.
  - If argmax == next draft token, ACCEPT, output, continue.
  - If mismatch, ACCEPT current logit's argmax (it's still a valid output), STOP.
- **Savings**: zero on the verified tokens (we still ran forward passes), but we avoid one SAMPLE step per accepted token. **Wait this is misleading — verify-by-rerun = no savings**.

**SCOPE A as written gives no actual speedup.** PLD's win comes from BATCHED verify. Skip Scope A.

## Implementation Plan — SCOPE B (Real, multi-token batch)
**Goal**: 1.5-2x on code, 1.0x on chat.

### B.1 Batched forward pass (the big one)
Refactor the per-token loop to optionally process `n_batch ≥ 1` tokens at once. New code path coexists with single-token path.

**Files affected**:
- `engine/runtime/winmoe_inference.c` — main loop
- `engine/runtime/attention.h` — gqa_attention needs to handle multi-position queries
- `engine/runtime/deltanet_impl.h` — DN state update for batch
- `engine/runtime/winmoe_inference.c` MoE block — token-grouped expert dispatch

#### B.1.a Embedding (easy)
Tokenize → embedding lookup for all `n_batch` positions. Output: `[H * n_batch]` hidden states.

#### B.1.b GQA attention (medium)
Currently single-position. Need:
- Q for batch: `Q[n_batch * nqh * hd]`
- KV cache: APPEND n_batch new K/V rows.
- Attention: each batch position p attends to KV cache positions [0..base_pos+p].
- Softmax + weighted sum.
- llama.cpp does this trivially (it uses `n_tokens > 1` mode for prompt). We need to mirror.

#### B.1.c DeltaNet (HARD)
DN is recurrent. State updates SEQUENTIALLY per token. For batch:
- Process token 0: q,k,v → update state → output 0
- Process token 1: q,k,v → update state → output 1
- ...
This is just the existing per-token path applied n_batch times INSIDE one layer. No real parallelism, but no big change either.

#### B.1.d MoE batched (HARD)
For each batch position, router selects K experts. Across `n_batch` positions, total K*n_batch routing decisions. To save IO:
- Build `expert_to_positions[]` map: for each unique expert ID, list of positions that selected it.
- For each unique expert: read weights once (cache miss → 1 SSD read), compute output for ALL positions that route to it.
- Add weighted contributions back to each position's `moe_out`.
- **IO savings ratio = unique_experts / (K * n_batch)**. For K=10, n_batch=4, typical 25-30 unique experts vs 40 total positions = ~30-40% IO reduction.

#### B.1.e LM head (easy)
Run AVX2 q6k_matvec for each position. `n_batch * 600 ms / 4 cores` (already OMP).

### B.2 Speculative decode loop
After base-pass logits, verify draft:
- For each batched position p (corresponds to draft token p): argmax of logits[p]. If matches draft[p], advance accepted_count++.
- Output the accepted prefix. Discard logits past the rejected position.
- Use the LAST accepted position's logits as the next sampling step (it's "free" — already computed).

### B.3 Fallback
If n-gram lookup returns 0 drafts, run normal single-token path (no batch overhead).

## Implementation Effort Estimate
- A.1 n-gram matcher: 1 hour (small isolated module)
- B.1.a embedding for batch: 1 hour
- B.1.b GQA batched: 4-6 hours (need to refactor KV cache, attention loop)
- B.1.c DN sequential per batch: 1 hour (just call existing per-token N times)
- B.1.d MoE token-grouping: 6-8 hours (router refactor, expert dispatch refactor)
- B.1.e LM head batched: 1 hour
- B.2 speculative loop integration: 2 hours
- Tests: 4 hours (compare batch vs single-token output, check accept rates)
- Total: **~24-32 hours** of focused work. Two solid sessions.

## Critical Files Reference

| File | Change scope |
|---|---|
| `engine/runtime/pld.h` (new) | n-gram matcher |
| `engine/runtime/winmoe_inference.c` | main loop + speculative decode + token-grouped MoE |
| `engine/runtime/attention.h` | gqa_attention multi-position |
| `engine/runtime/deltanet_impl.h` | document sequential per-batch usage |

## Verification

1. **Single-token equivalence**: with `n_batch=1`, output must be bit-identical to current path.
2. **Batched correctness**: with `n_batch=4` and FIXED draft tokens (no PLD lookup), the LOGITS at each position must match what we'd get from 4 sequential single-token passes.
3. **PLD acceptance**: on fibonacci prompt 200 tokens, measure acceptance rate. Target >= 1.3 effective tokens per forward pass.
4. **Coherence regression**: chat "Hello" prompt still produces canonical reply.
5. **Code quality**: fibonacci output is still valid Python.

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| KV cache batch refactor introduces position-mapping bug | Medium | Single-token equivalence test (item 1 above) |
| Token-grouped MoE has subtle accumulation bug | Medium | Diff each layer's `ffn_out` against single-token path |
| n_batch overhead exceeds savings on chat workload | High on chat | Skip PLD entirely if context has no good n-grams (cheap check) |
| Worse cache eviction (more experts touched per batched call) | Medium | Tune n_batch based on cache headroom |
| Memory pressure (n_batch × hidden + n_batch × KV cache) | Low | Cap n_batch=4; small fraction of model RAM |

## Open Questions to Resolve Before Coding
1. Does Qwen3.5's MoE router NEED to see all batched positions at once, or can we route each independently? (Should be independent — router is pointwise.)
2. Is DN's state actually per-batch-element OR per-sequence? (Per-sequence — same state across all batched tokens.)
3. Can we keep the AsyncRead infrastructure for batched expert reads? (Yes — issue all unique expert reads in one phase.)
4. What's the ideal n_batch given our SSD QD limits?

## What I Need The ML Expert To Critique
- Whether 1.5-2x speedup target is realistic for this engine architecture
- Whether token-grouped MoE batching is the RIGHT mechanism, or if there's a smarter approach (e.g., expert weight reuse across batch elements without full grouping)
- Are there published benchmarks of PLD on hybrid Gated-DeltaNet + sparse MoE models?
- Should we use a smaller draft model (n-gram is zero-cost; tiny LM might be better) — recall earlier critique that 35B-as-draft fails due to SSD contention
- The 0% chat-workload speedup — is there any way to make PLD useful on creative text generation, or should we just gate it on detected code-like prompts?
- What's the floor on n_batch where overhead doesn't dominate? (n_batch=2 vs 4 vs 8?)
