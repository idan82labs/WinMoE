# WS4 — KV Compression Scenarios

## Objective
Model how aggressive KV-cache compression changes the RAM/VRAM budget available to expert caching.

## Why it exists
Recent KV-compression work suggests that a 4x–5x-class reduction in KV burden may be plausible in some settings. For this project, that matters primarily because recovered memory can potentially be reassigned to expert hotsets or staging buffers.

## Questions answered
- how much RAM/VRAM does KV occupy across target context/batch regimes?
- what cache budgets become possible under plausible KV-compression factors?
- does expert-cache feasibility materially change under those scenarios?

## Required outputs
- baseline KV budget estimates
- compressed-KV scenario table
- simulator reruns under adjusted memory budgets
- recommendation on whether KV compression is first-order enough to warrant dedicated engineering in this project

## Exit criteria
The final V2 can say clearly whether KV compression changes feasibility materially or only incrementally.
