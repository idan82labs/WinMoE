# parity-coding loop — WinMoE teacher-forced logit parity vs llama.cpp

## Purpose
Prove (or disprove) that WinMoE's logit distribution on coding prompts agrees with llama.cpp-397B within the thresholds defined below. No speed work happens until this loop passes.

## FROZEN GRAPH (non-negotiable this sprint)
- **Checkpoint**: `D:/models/qwen35-397b-q4/Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-*.gguf`
- **WINMOE_TOPK**: `10` via env (override the v10.47 K=8 hard-cap — do not change source yet)
- **Sampler**: argmax only (no sampling, no temperature)
- **Tokenizer**: via `llama-trace-dump --tokenize`
- **Disabled**: PLD, speculative, `WINMOE_GPU_KERNEL` alternates, `WINMOE_DN_CPU`, `WINMOE_GQA_CPU`, LRU aggressive tuning
- **Defaults**: cache 20 GB, `WINMOE_GPU_EXPERTS=200`
- **No mid-sprint env flips.**

## Pass Gate (all 5 prompts must satisfy)
- `top1_agreement ≥ 90%` over 100 positions
- `top5_overlap ≥ 0.7` over 100 positions
- `correct_token_rank ≤ 3` at ≥ 95% positions
- `correct_token_logit_delta_winmoe ≤ 1.5 × correct_token_logit_delta_llama` at ≥ 90% positions
- q6k AVX2-vs-scalar parity passes (max_abs_diff < 1e-3, rmse < 1e-4)

## Preflight — §1.6 AVX2 q6k parity
```
cd engine/runtime/test
# build q6k_parity.cpp with same flags as winmoe
# run against 7 real activations × 1007 rows
# PASS if thresholds met
```

## Main benchmark invocation
```bash
# 1. Generate reference prefixes (once per prompt)
cd D:/llama-cpp-src/build-cuda/bin
./llama-trace-dump.exe -m <397B_model> -ngl 0 \
  --tokens "<prompt_ids>" --dump-logits "<out_prefix.bin>" \
  --n-gen 100

# 2. Run WinMoE on the same prompt + same forced prefix
cd engine/runtime
WINMOE_TOPK=10 WINMOE_DUMP_ALL_LOGITS=<winmoe_out.bin> \
  ./winmoe.exe --model <397B_model> \
  --prompt-tokens "<prompt_ids_plus_prefix>" --tokens 100

# 3. Compute metrics + plots
cd loops/parity-coding
python harness.py --prompt=<N> --llama-logits=<llama_out.bin> \
  --winmoe-logits=<winmoe_out.bin> --output=results/<date>-prompt<N>/
```

## Trial cycle
1. Build both binaries (winmoe + llama-trace-dump).
2. Preflight q6k parity.
3. For each of 5 prompts: generate reference prefix → run WinMoE → compute metrics.
4. Inspect 4 drift plots per prompt.
5. If all prompts pass gate → Phase 2. If any fail → classify shape via §1.4 → diagnose via §1.8.
6. Log every run in `results.tsv`.

## Prompts (see §1.5 of plan)
1. Bracket + quote closure: `config = {"name": "server", "port": 8080, "handlers": [`
2. Indentation-sensitive Python: `def parse_tree(node):\n    if node.left:\n        result = parse_tree(node.left)\n        if result is None:`
3. Variable-name disambiguation: `def process(items, filters, results):\n    for item in`
4. Operator choice: `if user.age`
5. JSON output: `Write valid JSON for a user: {"name":`

All wrapped in `<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n`.
