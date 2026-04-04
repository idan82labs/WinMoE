# Autoresearch Loop: Get Coherent Output from WinMoE

## Goal
Model produces coherent English text matching llama.cpp quality.
Currently: "VverVVV.VAVV..." (single chars). Target: actual sentences.

## Baseline
- llama.cpp: "Thinking Process: 1. **Analyze" (coherent)
- Our engine: "Cancel..:V..ext" (Hello prompt) / "VverVVV.VAVV" (quantum prompt)
- <think> logit: -0.13 to -0.68 (should be top-1 or top-5)
- Peak logit: 12.9 (llama.cpp: 15.1)

## Hypothesis Queue
1. DeltaNet state not accumulating context across tokens — conv1d or state issue
2. The hard sigmoid in SwiGLU differs from exact sigmoid
3. The GQA KV cache position tracking may be wrong for prompt tokens
4. Expert FFN output scaling may differ from llama.cpp
5. The DeltaNet output norm + gate may have wrong ordering vs reference
6. Conv1d may need dt_bias added before SiLU (check reference)

## Trial Log
| Trial | Change | Result | Keep/Reject |
|-------|--------|--------|-------------|
| T0 (baseline) | v10.18 all fixes | VverVVV.VAVV (quantum) | baseline |
| T1 | CPU-only (no GPU) | "ami ID!! …M stage0\nLn,!" | FINDING: GPU vs CPU differ! CPU more varied |
| T2 | Exact sigmoid (replace hard sigmoid in SwiGLU) | "Aext....V....V...." same pattern | reject — no improvement |
| T3 | Revert conv1d kernel reversal (k=0→newest) | "_DESTROY_clientext.row.Cancel" WORDS! | **KEEP** — huge improvement! |
| T4 | Hello prompt with T3 fix | V,spaces,V,A,ver — mostly single chars | neutral — quantum prompt was better |
| T5 | Remove /fp:fast | AAAVAAAAAAVA — worse, all single chars | **reject** — /fp:fast is better |
| T6 | CPU-only + conv1d fix | "\nni!pc a日stage[\"i03" mix | reject — T3 GPU was better |
| T7 | K=4 + conv1d fix | "GAGGG challengilicki" | neutral — different words, <think>=+0.12 |
| T8 | No <think> in prompt | ".A.A.AVV" single chars | reject — <think> in prompt helps |

## Best Config: T3
- GPU enabled, conv1d UNreversed (k=0→newest), /fp:fast, K=10
- <think>\n in prompt
- Output: "_DESTROY_clientext.row.Cancel"
- <think> logit: -0.02 (almost zero!)

| T11 | CPU-only with ALL fixes | "\nni!pc a日stage" same as T6 | reject — GPU actually better |

## Key Findings from Web Research
1. GGUF converter transforms A_log → -exp(A_log) before storing (line 7633 of convert_hf_to_gguf.py)
2. Our decay formula exp(ssm_a * softplus(...)) IS correct — ssm_a is already -exp(A_log)
3. flash-linear-attention confirms: state *= exp(g), not state *= g (raw)
4. HuggingFace ref: g = -A_log.exp() * softplus(a + dt_bias), then state *= exp(g)
5. llama.cpp: gate = ssm_a * softplus(...), then state *= exp(gate) — SAME formula
6. Token 198 (\n) has logit 3.17 in ours vs 15.14 in llama.cpp — 12 point gap

## Remaining Gap Analysis
- Our logits for meaningful tokens (\n, Here) are 10-12 points BELOW llama.cpp
- Our logits for single-char tokens (V, A, ver) are SIMILAR to llama.cpp's
- The model correctly downranks garbage but fails to uprank correct tokens
- This is a SIGNAL problem, not a NOISE problem

| T12 | Reversed conv1d (ggml matching) | ".A.AVV.." single chars worse | reject — unreversed still better |
| T13 | Seed "The" after <think>\n\n | ".V.V.V.VVV." single chars | reject — seeding doesn't help |

## Verified Against 4 Sources
All 5 DeltaNet recurrence steps verified identical:
1. HuggingFace modeling_qwen3_5.py
2. flash-linear-attention naive.py
3. llama.cpp delta-net-base.cpp
4. GGUF converter: A_log → -exp(A_log)

## Conv1d Direction Paradox
- ggml_ssm_conv: w[3]*newest + w[0]*oldest
- Our unreversed (T3 best): w[0]*newest + w[3]*oldest
- Our reversed (T12): w[3]*newest = matches ggml BUT gives worse output
- Conclusion: our buffer ordering flipped relative to ggml, so unreversed compensates

## Session 2 Trials (v10.25-v10.26)

| Trial | Change | Result | Keep/Reject |
|-------|--------|--------|-------------|
| T14 | Subtract 1 from all norms (except ssm_norm) | h_rms EXPLODES (0.01→762K) | **reject** — GGUF stores correct values |
| T15 | Fix GPU SwiGLU: hard sigmoid → exact sigmoid | peak logit 8.9→10.1 (+13%) | **KEEP** — GPU was wrong activation |
| T15b | CPU-only experts (no GPU cache) | peak logit drops to 7.7 | reject — GPU experts are better |
| T17 | Conv1d ggml direction (k=0→oldest) | peak logit 10.1→8.8 | reject — paradox persists |
| T19 | Disable conv1d entirely | peak logit ~8.9, generates \n instead of 53 | neutral — conv1d not the bottleneck |
| T20 | Compile without /fp:fast | peak logit drops to 6.0 | reject — /fp:fast helps |
| T21 | V head tiled reorder (h%16 vs h/4) | peak logit 9→4.3 | reject — GGUF uses grouped order |

## GGUF Metadata (verified)
- key_length=256, value_length=256, rope_dim=64
- full_attention_interval=4, ssm.group_count=16, ssm.time_step_rank=64
- rope.dimension_sections=[11,11,10,0] (YaRN sectioned)
- No routed_scaling_factor

## Confirmed Correct
- DeltaNet scale: 1/sqrt(128) ✓ (6 sources)
- DeltaNet algorithm: norm→SiLU(gate)→proj ✓ (3 sources)
- QKV split: Q(2048)+K(2048)+V(8192) ✓
- Q+Gate deinterleave: [Q_h0(256),Gate_h0(256),...] ✓
- RoPE: split-halves, partial 0.25, norm-then-rope ✓
- V head grouping: h/4 (grouped, not tiled) ✓ (empirical)
- All GGUF dimensions: match model config ✓

## STATUS: GPU SwiGLU fixed, ~6-point gap remains (peak ~9 vs target ~15)
- Run-to-run variance: ±1 point due to /fp:fast + OpenMP
- Disabling conv1d has NO significant effect → gap is not from conv1d
- /fp:fast gives +3 points over strict FP → accumulated precision matters
- Next step: build llama.cpp with CUDA for layer-by-layer numerical comparison
- Or: write Python reference implementation for single-layer validation

## Option B ready if needed
Fall back to using llama.cpp for DeltaNet computation, our engine for MoE speed.
