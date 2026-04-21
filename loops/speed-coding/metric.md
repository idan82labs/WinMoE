# Metrics

## Primary
- **tok_s_endtoend**: total tokens (prompt + generated) / wall-clock seconds. Read from `Tokens: N in M ms = X tok/s` line in stderr.
- **tok_s_steady**: average ms/tok over generation tokens 5..end (skip warmup variance). Computed from per-token Token lines.

## Guardrails (any regression = reject)
- **coherence_chat**: bench A must produce `<think>\n\n</think>\n\nHello! How can I help you today?<|im_end|>` (decoded). Pass/fail.
- **think_logit**: at tok=8 of bench A, `Token 248068 (<think>) logit = X` from stderr. Must be ≥ 18 (current baseline 21 — allow ~15% drift).
- **code_smoke**: bench B output decodes to valid Python (compiles via `ast.parse`). Pass/fail.
- **vram_safe**: GPU VRAM_used line must report ≤ 7.5 GB.

## Component diagnostics (info, not gates)
- attn_ms (per token avg)
- expert_ms
- lm_head_ms
- router_ms
- cache_hit_rate (RAM expert cache, %)
