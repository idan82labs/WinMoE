# Decision Record: 397B First Success

Date: 2026-03-25
Status: FROZEN CHECKPOINT

---

## What happened

Qwen3.5-397B-A17B was successfully loaded and executed on a consumer Windows laptop using llama-cpp-python with a GGUF IQ2_XXS quantization. Three K values were tested (10, 7, 5) producing a clean speed-vs-K tradeoff curve.

## Proven facts

1. **397B local execution is now proven.** A 397-billion parameter MoE model runs on consumer Windows hardware (40 GB RAM, 8 GB VRAM, Samsung 980 NVMe). The model generates coherent English text at usable speeds.

2. **K sweep direction matches theory.** The measured 2.3x speedup from K=10 to K=5 is consistent with the V2 spec's prediction that expert loading is the dominant bottleneck (I/O-bound regime). The relationship is approximately linear: halving K roughly doubles tok/s.

3. **GGUF metadata patching is a valid K control mechanism.** Changing `expert_used_count` in the first GGUF shard's header at byte offset 1425 successfully controls the number of active experts in llama.cpp inference.

4. **The system is viable at 0.5-1.0 tok/s.** This is usable for interactive applications (comparable to early ChatGPT speeds).

## NOT proven

1. **Quality at K=5 is not established.** No perplexity measurement, no benchmark evaluation, no extended coherence test. The single 50-token output at K=10 was coherent, but K=5 output quality is completely unevaluated.

2. **IQ2_XXS quantization quality is not evaluated.** 2-bit quantization is known to significantly degrade model quality. Results should be treated as infrastructure validation, not quality benchmarks.

3. **Reproducibility across sessions.** Only tested in a single session. OS page cache state, background processes, and thermal throttling could affect results.

## Baseline preservation

The following must be preserved before any optimization experiments:

- GGUF files at `D:/hf_cache/models--unsloth--Qwen3.5-397B-A17B-GGUF/`
- Baseline JSON at `state/baselines/397b_k5_cpu_iq2.json`
- K sweep data at `experiments/k_sweep/results.tsv`
- Run script at `work/baseline/run_397b.py`
- Patch script at `work/baseline/patch_k_experts.py`

**The GGUF file is currently set to K=7 (last test run).** Restore to desired K before next experiment.

## Decision

Freeze this as the project's first empirical checkpoint. All subsequent optimization experiments must record their results relative to this baseline. Any change that regresses below 1.0 tok/s at K=5 without quality justification should be rejected.
