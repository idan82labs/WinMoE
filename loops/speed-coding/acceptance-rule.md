# Acceptance Rule

## Keep
- Primary tok_s_endtoend improves by **≥ 5%** vs current baseline AND
- All guardrails pass (coherence + think_logit + code_smoke + vram_safe).

## Reject
- Primary regresses by > 5%, OR
- Any guardrail fails.

## Neutral
- Primary within ±5% AND all guardrails pass.
- Neutral changes are kept if they are correctness-improving (e.g., race fixes) or unblocking (e.g., infrastructure).

## Protocol
- After accept: update baseline reference, commit to git with `v10.X: <change> — <tok_s>`.
- After reject: revert change, document the negative result in notes.md.
- Mid-trial: if PC RAM drops below 5 GB free, abort trial (memory thrashing invalidates speed measurements).
