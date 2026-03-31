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

## Next Hypotheses to Try
1. The GPU Q8_0 kernel shared memory accumulation may lose precision vs CPU
2. Double-precision state recurrence (use scalar double instead of AVX float)
3. The DeltaNet output scale 1/sqrt(128) may need to match llama.cpp exactly
4. Check if we need to add dt_bias to the alpha BEFORE softplus in conv1d context
