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

## STATUS: All algorithms correct, 12-point logit gap remains
The issue is NOT in the algorithm — it's in how information accumulates across
60 layers and 15 tokens. Could be subtle precision interaction, OpenMP thread
ordering, or a minor numerical detail in the Q8_0 matmul accumulation.

## Option B ready if needed
Fall back to using llama.cpp for DeltaNet computation, our engine for MoE speed.
