# Loop Runner Spec

## Purpose

Define how local autoresearch-style loops should be executed.

## Minimal loop contract

A loop runner must:
1. read target.md
2. read metric.md
3. read acceptance-rule.md
4. read rollback-rule.md
5. inspect the latest baseline in baseline.md
6. run exactly one bounded trial
7. append one row to results.tsv
8. append or update notes.md
9. if accepted, update baseline.md
10. if the lesson generalizes, add it to broader memory
