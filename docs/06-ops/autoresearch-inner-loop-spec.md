# Autoresearch-Style Inner Loop Spec

## Purpose

This spec adds a Karpathy/autoresearch-style experimental inner loop to the Flash-MoE Windows V2 workspace.

The goal is to make each major workstream executable as a tight, measurable, revertable improvement loop rather than only as a broad project phase.

The loop is intentionally simple:
1. pick one narrowly scoped target surface,
2. define one canonical benchmark or evaluation command,
3. make one concrete change,
4. run the benchmark,
5. compare against the current baseline,
6. keep or discard the change,
7. record the lesson in memory.

## Why this exists

The broader V2 filesystem already supports:
- phases,
- gates,
- workstreams,
- agents,
- evidence tracking,
- and synthesis.

What it previously lacked was a small inner optimization loop similar to Karpathy's autoresearch pattern:
- one mutable target,
- one benchmark command,
- one score ledger,
- one accept or reject rule,
- one rollback rule,
- and one habit of iterating with very small, testable deltas.

This layer is meant to sit inside the existing workstreams, not replace them.

## Core loop invariants

Every loop must define:
- target surface,
- canonical evaluation path,
- primary metric,
- secondary guardrails,
- accept or reject rule,
- rollback rule,
- results ledger.

## Project-specific adaptation

For Flash-MoE Windows, the most useful loop surfaces are:
1. I/O path
2. Timing harness
3. Simulator policy evaluation
4. Layerwise K_l schedule search
5. Predictor or partial-overlap experiments

## General execution pattern

1. Confirm baseline exists and is reproducible.
2. Make one small change only.
3. Run the canonical evaluation command.
4. Record change, command, metric, guardrails, and decision.
5. If accepted, update baseline and lessons.
6. If rejected, revert working change and record why.
7. Repeat.

## Required outputs

Each loop directory must contain:
- target.md
- metric.md
- runbook.md
- results.tsv
- notes.md
- baseline.md
- acceptance-rule.md
- rollback-rule.md

## Success condition

This layer is working correctly if experiments are easy to rerun, each loop has a stable baseline, improvements are accepted or rejected quickly, and lessons accumulate without bloating the global project memory.
