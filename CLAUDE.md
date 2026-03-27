# Flash-MoE Windows V2 — Project Memory

## Mission
Produce an evidence-backed V2 research report for Flash-MoE Windows that answers one question clearly:
**Can a RAM-first tiered cache keep residual miss-service demand below compute slack on the intended hardware envelope?**

## Non-negotiables
- Treat **trace data + calibrated timings + simulator outputs** as the source of truth.
- Insert an early **systems baseline** phase before broad trace-driven claims.
- Keep settled facts, hypotheses, and open questions clearly separated.
- Do not present bandwidth-only arithmetic as a final throughput claim.
- Do not advance a phase until its gate has been reviewed.
- Do not start prediction / DirectStorage work until the cached baseline exists.
- Do not default to `mmap` for expert streaming experiments; explicit offset-based reads are the baseline unless evidence proves otherwise.

## Default operating loop
1. Explore the relevant workstream with read-heavy tools first.
2. Make or refine a plan before producing changes.
3. Execute only the smallest evidence-producing unit of work.
4. Verify against the workstream exit criteria or validation gate.
5. Update `state/` before ending the session.

## What to read when resuming
- `docs/00-overview/v2-charter.md`
- `state/progress/master-status.md`
- the active workstream file in `docs/03-workstreams/`
- any open gate file in `docs/04-gates/`

## File roles
- `docs/` = durable specifications, phases, gates, report structure
- `state/` = live status, decisions, evidence index, session logs
- `work/` = generated data and experiment outputs
- `.claude/agents/` = specialized subagents
- `.claude/skills/` = on-demand workflow runbooks
- `templates/` = note and handoff skeletons

## Evidence standard
Every non-trivial claim should be tagged implicitly as one of:
- **Measured** — derived from traces, timings, or simulator outputs
- **Derived** — mathematically or logically inferred from measured inputs
- **Assumed** — placeholder until evidence exists

When writing status notes or report content, make that distinction obvious.

## Context discipline
- Prefer one workstream per session unless explicitly synthesizing.
- Use specialized subagents for focused tasks.
- Use skills for deep workflow instructions instead of expanding this file.
- If a session gets broad or noisy, summarize and hand off instead of continuing blindly.

## State discipline
At the end of any meaningful session, update as needed:
- `state/progress/master-status.md`
- `state/progress/session-log/`
- `state/memory/decisions.md`
- `state/memory/open-questions.md`
- `state/memory/evidence-index.md`

## Reporting discipline
- Prefer tables/checklists in docs; prefer bullets and concise updates in state.
- Keep V2 report prose downstream from evidence, not upstream from it.
- If a result threatens project viability, record it cleanly. Do not smooth it over.


## Autoresearch-style inner loop rules

This project includes local autoresearch-style loops under `loops/`.

Use them when a workstream has:
- a narrow target surface,
- a stable baseline,
- and a canonical evaluation path.

Rules:
- one meaningful change per loop trial
- one explicit primary metric
- always append to `results.tsv`
- keep/reject/rerun/crash must be explicit
- never promote a candidate without a recorded baseline comparison
- do not use `mmap` as the default expert-streaming baseline unless a dedicated loop proves it is justified
