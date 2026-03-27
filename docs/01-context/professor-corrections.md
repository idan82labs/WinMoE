# Professor Corrections

## Hard corrections already adopted
- Use full per-layer expert-demand bytes, not per-expert bytes, in throughput arithmetic.
- Treat causal decode as the conservative baseline on Windows unless prediction creates real lead time.
- Focus the project on RAM-first cache feasibility, not zero-cache overlap optimism.
- Fit affine timing models before making strong tok/s claims.

## Follow-on implications now added
- Add a systems-baseline phase before full trace/simulator work.
- Judge static hotset trustworthiness by boundary stability and cut-off drift, not only global stationarity.
- Use static vs recency vs oracle gap curves to decide where adaptivity belongs.
- Use service-demand distributions, not only hit rates, to define the cache cliff.
