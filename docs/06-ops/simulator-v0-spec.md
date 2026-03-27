# Simulator V0 Spec

## Inputs
- routing traces or derived trace summaries
- timing fits for SSD and RAM-to-GPU at minimum
- cache capacities
- policy type

## Outputs
- miss bytes by stage
- request counts by stage
- estimated miss-service demand distribution
- mean and high-quantile safety margins vs compute slack
- policy gap comparisons

## Minimum supported policies
- static hotset
- adaptive recency buffer
- static + adaptive hybrid
- oracle reference if practical

## First-order success criterion
The simulator should be able to tell whether the cached baseline is likely to stay below compute slack often enough to be viable.
