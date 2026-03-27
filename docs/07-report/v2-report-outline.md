# V2 Report Outline

## 1. Revised problem framing
Why the project is now a RAM-first feasibility study.

## 2. Systems-baseline bring-up lessons
Why correctness, explicit I/O primitives, cache/no-cache behavior, and slab layout were treated as prerequisites.

## 3. Corrected theoretical baseline
Corrected byte accounting, causal model, and why zero-cache optimism broke.

## 4. Calibrated timing model
Why affine timing fits replace bandwidth-only arithmetic.

## 5. Trace statistics that determine cacheability
Popularity, stability, working-set growth, overlap, reuse distance.

## 6. Policy comparison and tier sizing
Static vs recency vs hybrid vs oracle. Critical cache size and hybrid split.

## 7. KV-compression scenarios and layerwise `K_l`
How reduced KV burden and active-expert reduction change feasibility.

## 8. Prediction and DirectStorage
Whether either still has enough headroom to matter after the cached baseline.

## 9. Final recommendation
Go / conditional-go / no-go with explicit assumptions.
