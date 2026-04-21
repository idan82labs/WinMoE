# speed-coding loop — WinMoE → Code-Capable + 1+ tok/s

## Purpose
Drive WinMoE from 0.22 tok/s coherent chat to ≥1 tok/s sustained on coding workloads, without regressing quality.

## Standard benchmark commands

### Bench A — coherence regression (chat)
Standard "Hello" chat prompt. Catches quality regressions early.
```bash
cd engine/runtime
PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin:$PATH" \
WINMOE_TRACE_OUT="/c/Users/idant/loop_trace.txt" \
OMP_NUM_THREADS=2 MSYS_NO_PATHCONV=1 timeout 600 ./winmoe.exe \
  --model "D:/models/qwen35-397b-q4/Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-00001-of-00006.gguf" \
  --tokens 30 > "/c/Users/idant/loop_run.log" 2>&1
```

Expected output (after EOS): `<think>\n\n</think>\n\nHello! How can I help you today?<|im_end|>`

### Bench B — code generation (200 tokens, code prompt)
Once `--prompt-tokens` is wired, run with `def fibonacci(n):` tokenized prompt × 200 tokens.

## Build
```bash
cmd //c "C:/Users/idant/build_winmoe.bat"
```

## Decode tokens for verification
```bash
cd "D:/llama-cpp-src/build-cuda/bin" && \
  PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin:$PATH" \
  ./llama-trace-dump.exe -m "D:/models/qwen35-35b-q4/Qwen3.5-35B-A3B-Q4_K_M.gguf" -ngl 0 \
    -o "/c/Users/idant/decoded.txt" \
    --tokens "<comma_sep_ids>" --filter "DECODE_ONLY" > /dev/null 2>&1
grep "^DECODED" "/c/Users/idant/decoded.txt"
```

## Trial cycle
1. Apply ONE change (one optimization knob).
2. Build clean (kill stale winmoe).
3. Run Bench A — check coherence + read tok/s.
4. Run Bench B — check code quality + read tok/s.
5. Append row to results.tsv (run python run_trial.py for automated path).
6. keep / reject / neutral based on acceptance-rule.md.
7. Commit if accepted: `git commit -m "v10.X: <change> — <result tok/s>"`.

## Optimization queue (from elegant-singing-koala.md plan)
- T1: Phase 0 baseline — establish current state on chat + 5 code prompts
- T2: Phase 2 cross-stream race fixes (2 sites in gpu_offload.cu)
- T3: Phase 3 GPU expert LRU + bump to 200
- T4: Phase 4a OMP DN state recurrence
- T5: Phase 4b cache-blocked AVX2 Q6_K LM head
- T6: Phase 5 Prompt-Lookup Decoding
- (Phase 1 logit fix is conditional on Phase 0 results)
