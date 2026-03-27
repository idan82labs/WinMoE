# Claude Code Best Practices Applied to This Filesystem

This repository structure deliberately mirrors Claude Code guidance.

## 1. Verify work
The project is structured around evidence-producing workstreams, validation gates, and simulator outputs. No phase is considered complete on intent alone.

## 2. Explore first, then plan, then code
`CLAUDE.md`, the skills, and the workstream files all enforce an explore -> plan -> execute -> verify loop.

## 3. Keep CLAUDE.md short and durable
The root `CLAUDE.md` is intentionally compact. It covers project-wide operating rules only. Deeper instructions live in skills and workstream specs.

## 4. Use skills for on-demand workflows
The `.claude/skills/` directory contains runbooks for kickoff, trace acquisition, timing calibration, simulator work, gate review, and synthesis.

## 5. Use focused subagents
Project agents have narrow responsibilities and limited tools where possible. The goal is context isolation and predictable behavior.

## 6. Use hooks to leave durable state
The project-level hooks are simple but useful: they initialize session logs, record file-edit events, and prompt a state update on stop.

## 7. Manage context aggressively
The repository separates:
- durable project docs
- live project state
- generated experiment outputs

This prevents every session from carrying the full project narrative in the live chat context.

## 8. Prefer testable claims over optimistic prose
The structure is built to reduce the risk of mistaking planning quality for research progress.
