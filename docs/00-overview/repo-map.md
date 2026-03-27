# Repository Map

## Durable planning docs
- `docs/00-overview/` — charter, applied best practices, repo map
- `docs/01-context/` — baseline assumptions, professor corrections, risks
- `docs/02-phases/` — phase plan and progression model
- `docs/03-workstreams/` — execution specs per workstream
- `docs/04-gates/` — validation gates
- `docs/05-workload/` — team topology, workload split, cadence
- `docs/06-ops/` — operational specs for traces, timings, simulator
- `docs/07-report/` — final V2 report structure and evidence mapping

## Live state
- `state/memory/` — decisions, facts, open questions, evidence index
- `state/progress/` — master status, phase status, session logs

## Work products
- `work/traces/` — raw and processed routing traces
- `work/timings/` — raw measurements and fitted timing models
- `work/sim/` — simulator inputs, configs, outputs
- `work/validation/` — gate evidence and synthesized checks

## Claude Code integrations
- `.claude/agents/` — subagents
- `.claude/skills/` — reusable workflows
- `.claude/commands/` — shortcuts
- `.claude/settings.json` — hooks
