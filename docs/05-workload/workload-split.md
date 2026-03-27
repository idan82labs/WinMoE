# Workload Split

## Recommended primary streams
1. Systems baseline
2. Trace acquisition
3. Timing calibration
4. Simulator and policy comparison
5. Memory-budget and `K_l` scenarios
6. Prediction / DirectStorage evaluation
7. Report synthesis

## Sequencing rule
The first four streams are not equally parallel:
- systems baseline must start first
- timing calibration can begin in parallel once primitive-path candidates exist
- trace acquisition should follow the selected or justified runtime baseline
- simulator work should wait for at least one trace package and first timing fits

## Suggested ownership
- Stream 1: systems / correctness owner
- Stream 2: routing / instrumentation owner
- Stream 3: measurement / calibration owner
- Stream 4: simulator / policy owner
- Stream 5: memory-budget / `K_l` owner
- Stream 6: advanced levers owner
- Stream 7: synthesis owner


## Added shared workload - Autoresearch-style inner loops

Every major workstream should own at least one local loop once its baseline exists.

Priority order:
1. I/O path loop
2. Timing fit loop
3. Simulator policy loop
4. Layerwise K_l loop
5. Predictor value loop
