# Research Question: Residual Hidden-State Drift in Custom Qwen3.5-397B Inference

**Audience:** ML inference specialist or senior systems/numerics engineer familiar with MoE transformers, Gated DeltaNet, and quantized inference (GGML-family kernels).

**Ask in one sentence:** After fixing every algorithmic and data-layout bug we could find (and verifying layer-0 matches Python FP64 to 6 decimal places), our 60-layer forward pass still produces a hidden state whose direction systematically underweights the correct next-token by ~12 logit points versus llama.cpp — where should a skeptical reviewer look next?

---

## 1. System Under Test

- **Model:** Qwen3.5-397B-A17B, Q4_K_M quantized GGUF (6 shards, 228 GB). Also tested on Qwen3.5-35B-A3B (18 GB, 40 layers).
- **Hardware:** i7-11800H, RTX 3070 Laptop 8 GB VRAM, 40 GB RAM, Samsung 980 NVMe.
- **Our engine:** fully-custom C + CUDA, ~5000 lines; SSD-streams experts at run time because the model doesn't fit in RAM+VRAM.
- **Architecture (397B):**
  - 60 blocks = 45 Gated-DeltaNet + 15 GQA (every 4th)
  - hidden = 4096; DeltaNet heads = 64 value / 16 key × 128 dim (QKV = 2048+2048+8192 = 12288)
  - GQA heads = 32 Q × 256 × 2 (Q+gate concat) / 2 KV × 256
  - MoE = 512 experts, top-K=10 + 1 shared expert with sigmoid gate
  - Partial RoPE factor 0.25 (rope_dim = 64 of 256), IMROPE mode, sections=[11,11,10,0]
- **Quantization formats used:** Q4_K (expert gate/up), Q5_K (expert down), Q8_0 (all attention + embeddings), Q6_K (LM head). Q8_K as activation format for integer dot products.
- **Baseline for comparison:** llama.cpp CUDA build (`D:\llama-cpp-src\build-cuda`) on the same GGUF; architecture id `qwen35moe`, rope_type `LLAMA_ROPE_TYPE_IMROPE`.

## 2. Current Symptom (Quantitative)

With prompt `<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n` (token IDs `[248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198]`):

| Measurement | Our engine (397B) | llama.cpp (35B ref) |
|-------------|-------------------|---------------------|
| Peak logit at first generation step | ~10 | 14.39 |
| Logit std across vocab | ~2.1 | ~3+ (implied) |
| Logit for `<think>` (248068) | 1.7 | top-1 candidate |
| Logit for `\n` (198) | 3.5 | 14.39 |
| Logit for '.' (13) | **10.7 (winner)** | ~low |
| Logit for 'user' (846) at pos 0 prediction | -0.03 | expected top-1 (~15) |
| Generated tokens | '.', '://', 'import', ' \n', '.' | coherent English |
| Run-to-run variance (/fp:fast) | ±1 point | N/A |

Pattern: the model generates **real text tokens** (not garbage) but tokens that look like **pre-training/code distribution**, not chat-finetuned responses. Every prompt position has its correct next-token 8–12 points below whatever random common token wins.

## 3. Components Verified Correct (Against Python FP64 Reference)

Python validator scripts (in `engine/runtime/*.py`) read the raw GGUF, dequantize with the GGML reference layout, and compute in FP64. All match our C engine to 6 decimal places:

| Component | Verification method | Result |
|-----------|---------------------|--------|
| Q8_0 embedding lookup, token 248045 | Direct byte-level dequant | embedding[0..3] = 0.007545 0.007545 0.011526 0.007335 — **match** |
| RMSNorm layer-0 | FP64 recompute from embedding + attn_norm | normed[0..3] match to 1e-6 |
| DeltaNet QKV projection (Q8_0 × FP64) | Per-row dot product | qkv[0..7] match to 1e-6 |
| Router selection (FP32) | Full softmax top-10 | **exact match** — same experts 213/136/53 (old prompt), 413/206/153 (new prompt), same softmax weights to 4 dp |
| Q4_K expert gate output (expert 213, row 3) | Full dequant + dot product | gate[3] = 0.0336 (FP64 reference), 0.0497 (our Q8_K integer path) — difference explained by Q8_K activation quantization, the weight values themselves match GGML `dequantize_row_q4_K` |
| LM head Q6_K dot product | Manual `q6k_dot_block` call at runtime | Matches full-matvec logit to 0.01 |
| Attention RMSNorm (per-head q_norm, k_norm) | Weight magnitudes across layers | All ~0.7–1.5 RMS — no anomaly |
| GQA attention softmax (tok=7, head 0, L3) | Score array | `[3.19, 20.19, 14.96, 17.03, 18.91, 17.47, 18.53, 24.14]` — spread **21 points**, properly peaked on self + "user" token |
| All GGUF metadata | Dump + validate | dims match model config, including `rope.dimension_sections=[11,11,10,0]`, `rope.dimension_count=64`, `attention.key_length=value_length=256` |
| Q4_K / Q5_K / Q6_K struct sizes | `sizeof()` check | 144/176/210 bytes — no compiler padding |

## 4. Fixed Bugs (This Session, Each Verified by A/B Testing)

| Bug | Magnitude of fix | How found |
|-----|------------------|-----------|
| **Q4_K nibble layout** — we read 16 bytes per sub-block with same scale for both nibbles; GGML reads 32 bytes where low nibbles = sub-block A with scale[2k], high nibbles = sub-block B with scale[2k+1]. Half of every expert dot-product was multiplying wrong weight × wrong activation. | **+7 logit points** (peak 5 → 12) | Reading GGML `dequantize_row_q4_K` in `ggml-quants.c` |
| **Q5_K same nibble layout + high-bit mask** (bit `chunk*2` for low nibbles, `chunk*2+1` for high) | Included in +7 (expert DOWN projection) | Same place |
| **GPU SwiGLU used hard-sigmoid** `0.5 + 0.5x/(1+|x|)` while CPU used exact | +1–2 points | Code grep after prof asked about activations |
| **Wrong prompt token IDs** — used Qwen2.5 152K vocab IDs (`<|im_start|>`=151644) which map to Arabic text in Qwen3.5's **248K** vocab. Real IDs: `<|im_start|>`=248045, `<think>`=248068, `user`=846, `assistant`=74455 | Changes pattern of wrong tokens (from byte-level garbage → code-like tokens); didn't close the logit gap | GGUF tokenizer dump via Python `gguf` library |
| **Dynamic DeltaNet + GQA dimensions** from GGUF metadata (needed for 35B to run at all; 397B was hard-coded correctly) | 35B now loads | 35B had NaNs from hard-coded 64 value heads when actual is 32 |
| Misc: RMSNorm scalar-tail bug (never fired for 8-aligned dims), `feed_forward_length` parsing, struct-size checks | Minor / hygiene | Code review + professor consultation |

## 5. Hypotheses Tested and Ruled Out

| Hypothesis | Test | Result |
|-----------|------|--------|
| Conv1d kernel direction (weight[0] × newest vs oldest) | A/B | Both give ~same peak logit after Q4_K fix; disabling conv1d entirely gives nearly identical output — **not the dominant issue** |
| Subtracting +1 from norm weights (wrong sign of "+1 convention") | A/B | h_rms explodes 0.01 → 762K — confirmed the GGUF stores w+1 and our math is right |
| V-head tiled reorder (`h % 16` vs `h / 4`) | A/B | Tiled is WORSE on our GGUF — this file uses grouped order |
| IMROPE with sector filter | A/B + code study | Confirmed IMROPE for text-only = standard NEOX split-halves (all p_t=p_h=p_w=same text pos, all thetas advance uniformly). **Our RoPE is mathematically equivalent.** |
| Flat-softmax ("attention broken") | Direct score dump | Spread = 21 pts at L3 head 0 — **attention is peaked** |
| Q8_K precision loss (force float path) | Compile flag | Float path gives nearly identical output → quantization is not the issue |
| CPU-only experts (disable GPU cache) | A/B | CPU-only is **worse** than mixed — so GPU path isn't the culprit |
| `/fp:fast` vs `/fp:precise` | Recompile | `/fp:fast` is +3 pts better — unusual; suggests some numerical condition `/fp:fast` happens to handle better |
| DeltaNet disabled (zero `o_out` for DN layers) | A/B | Slightly worse, not dramatically — DeltaNet contributes some signal but isn't the missing piece |

## 6. Per-Layer Trajectory (tok=7, last prompt token, 397B)

| Layer | Type | attn_rms | moe_rms | h_rms (post-residual) |
|-------|------|----------|---------|-----------------------|
| L0 | DN | 0.75 | 0.02 | 0.74 |
| L1 | DN | 0.033 | 0.018 | 0.73 |
| L2 | DN | 0.025 | 0.028 | 0.75 |
| **L3** | **GQA** | **1.77** | **7.67** | **8.01** |
| L4 | DN | 0.22 | 25.8 | 28.5 |
| L5 | DN | 0.037 | 1.41 | 27.9 |
| ... | ... | (DN ~0.02–0.2, GQA ~1.5) | ... | grows to ~24 by L59 |

**DeltaNet contribution is 10–100× smaller than GQA** across most layers. MoE dominates magnitude by a large factor. Whether this is "correct" for Qwen3.5 we don't know — we can't get per-layer traces from llama.cpp on the 397B (can't load), and the 35B llama.cpp output got stuck in conversation mode despite `-no-cnv`.

## 7. Mathematical Framing of the Residual Drift

Let `h_ours` and `h_ref` be the final normed hidden states after 60 layers; let `W_i` be LM-head row for token i. We observe:

```
logits_ours[i]  = h_ours · W_i
logits_ref[i]   = h_ref · W_i
```

- For "noisy" garbage tokens (random high-ID tokens), our logits are CLOSE to reference magnitude.
- For the "correct" token (e.g., '\n'=198 after `assistant\n`), our logit is ~11 pts below reference.
- Peak logit in our engine is ~10 (some generic common token), reference is ~14.4 (the correct token).

Interpretation: `h_ours` differs from `h_ref` by a vector `e = h_ours − h_ref` that is:
1. **Persistently aligned** with common-token LM-head rows (so common tokens win)
2. **Anti-aligned** with chat/reasoning-specific LM-head rows (so `<think>`, `\n` lose)
3. **Deterministic** (same tokens win across runs)

This is NOT the signature of precision-only loss (which would look like random ε per dimension). It looks like `e` points in a **specific direction** — the direction that separates pre-training distribution from chat fine-tuning.

Across 60 layers, each layer contributes `δᵢ = layer_i_output_ours − layer_i_output_ref`. The sum `Σδᵢ = e`. For the drift to be directional and consistent, each `δᵢ` needs to have some coherent bias.

A multiplicative error (wrong scale) would attenuate magnitude uniformly. An additive error (wrong bias) would add a fixed vector. A layout error (wrong index matching) would look random. A missing activation nonlinearity would look structured.

## 8. Things We Have NOT Been Able To Verify

1. **Per-layer output after L0**: our Python FP64 reference is only built for layer 0 (embedding + attn_norm + QKV). Layer 1+ MoE is computationally expensive in pure Python.
2. **llama.cpp's internal hidden state for the exact same tokens**: llama.cpp's CUDA build can't load the 397B (231 GB VRAM request). The 35B should work but every `llama-cli` run ends up in conversation mode producing `>` prompts despite `-no-cnv`.
3. **Shared expert sigmoid gate values across layers**: we compute `sigmoid(dot(shexp_gate_inp, normed))` with double precision, but haven't verified `shexp_gate_inp` is a [hidden_dim] vector (single-output projection) vs something else.
4. **DeltaNet state accumulation over multi-token prompts**: we see L0 first-token output = 0.75 RMS, L1 first-token = 0.033 RMS. The 20× drop between L0 and L1 for the same token is unexplained — same code, same weights scale.

## 9. Candidate Root Causes Ranked By Our Current Suspicion

1. **Cumulative directional drift from FP32 accumulation order vs GGML's specific reduction pattern** — `/fp:fast` helps us by 3 points, which is suspicious.
2. **A Qwen3.5-specific detail we haven't implemented**: e.g., a token-type-aware mask, a specific attention scaling override (`hparams.f_attention_scale`), a YaRN frequency correction for even short sequences.
3. **Sign / indexing bug in DeltaNet state recurrence that only manifests for per-value-head states with empty K-group** — this would be consistent with L0→L1 drop.
4. **Z-gate (output gate) on DeltaNet**: we compute `normed × SiLU(z)` where z is from `attn_gate` Q8_0 projection. If z is systematically biased (e.g., always negative), SiLU(z) ≈ 0 and DeltaNet output is tiny. We haven't dumped z values.
5. **Numerical instability in softmax for attention scores with spread=21** — exp(-21) ≈ 7e-10, below denormal range on some paths; unlikely in practice.

## 10. Specific Questions for the Reviewer

1. **Given the pattern "wrong tokens are code-like, correct tokens are chat-like"**, does this suggest a specific layer-type is malfunctioning (e.g., pre-training → post-training specialization happens in particular blocks)? Is there a known architectural quirk in reasoning-tuned Qwen3.5 checkpoints that a naive implementation could miss?

2. **Is the 20× L0→L1 drop in DeltaNet `attn_rms` (0.75 → 0.033 for the first token) normal**, or does it suggest the state-recurrence isn't properly passing `beta * outer(k, v)` updates? We verified the delta-rule math against HF's `torch_recurrent_gated_delta_rule`, but empty-state edge cases are where we'd expect first-token bugs.

3. **For text-only IMROPE, does GGML's `ggml_mrope_cache_init` produce numerically identical cache to `ggml_rope_cache_init` when `p_t=p_h=p_w=same position` and all `theta_base_X` are equal**? Our reading of the code says yes; we want confirmation.

4. **Is there a canonical single-layer numerical reference** (a Python notebook, an ONNX snapshot, a known-good CPU implementation) for a Qwen3.5-MoE block that we could compile our first-layer Python reference against? The gap between what HF gives (FP64) and what llama.cpp gives (quantized integer) is 48% on a single expert dot product — which is "correct" for diagnosis?

5. **Given `/fp:fast` helps us 3 points**: what is the expected maximum swing between `/fp:fast` and IEEE-strict for a 60-layer transformer inference? In our experience it should be <0.5 pt; ours is 3. Does this alone tell you where to look?

6. **Is there a known "one-weird-trick" for Qwen-Next / Qwen3.5-MoE** (pre-norm vs post-norm convention, residual scaling factor, tied-weight trick) that differs from standard LLaMA/DeepSeek-MoE conventions?

7. **What is the single cheapest experiment we could run** that would eliminate the most hypotheses at once? Our hardware can't load the 397B in llama.cpp, but can run our engine and the 35B in llama.cpp simultaneously if we can get llama.cpp to exit conversation mode.

## 11. Artifacts Available for Inspection

- **Repo:** https://github.com/idan82labs/WinMoE (public, v10.39 at time of writing)
- **Key files:**
  - `engine/runtime/winmoe_inference.c` — main loop, ~1700 lines
  - `engine/runtime/deltanet_impl.h` — DeltaNet recurrence, ~350 lines
  - `engine/runtime/q4k_dequant.h`, `q5k_dequant.h` — fixed nibble layout
  - `engine/runtime/gpu_offload.cu` — CUDA kernels
  - `engine/runtime/attention.h` — GQA + RoPE + softmax
  - `engine/runtime/AUTORESEARCH.md` — trial log T0-T39
  - `engine/runtime/validate_layer0.py`, `verify_router.py`, `verify_expert.py` — Python FP64 reference scripts
- **Key commits to look at:**
  - v10.31 — the Q4_K/Q5_K layout fix (biggest win, +7 pts)
  - v10.34 — correct Qwen3.5 prompt tokens (was sending Arabic!)
  - v10.39 — current state with all diagnostics

---

**Tight ask:** if you can only read one file, read `engine/runtime/winmoe_inference.c` lines 800–1310 (one decoder block in our main loop) and `engine/runtime/deltanet_impl.h` end-to-end, then tell us what you'd verify next.
