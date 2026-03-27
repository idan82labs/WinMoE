# Gate 0B - Inner Loop Readiness

## Purpose

Ensure the project has a functioning autoresearch-style inner loop system before major optimization work begins.

## Requirements

The following loop directories must exist and be populated:
- loops/io-path-loop/
- loops/timing-fit-loop/
- loops/simulator-policy-loop/
- loops/k-schedule-loop/
- loops/predictor-value-loop/

Each loop must contain:
- target.md
- metric.md
- runbook.md
- baseline.md
- acceptance-rule.md
- rollback-rule.md
- results.tsv
- notes.md

At least the highest-priority loop (io-path-loop) must show one baseline run and one candidate run with a recorded keep or reject decision.
