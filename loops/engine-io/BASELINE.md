# Engine I/O Loop — Baseline

## mmap random reads (current llama.cpp approach)
- 586 MB/s on 3.7 MB expert blocks
- 5.96 ms per expert read
- Source: engine/io/explicit_io_bench.py

## explicit unbuffered reads (target)
- 2200 MB/s on same blocks
- 1.67 ms per expert read
- 3.6x improvement over mmap

## Implication
Per layer at K=3: mmap = 17.9 ms, explicit = 5.0 ms
Per token (60 layers): mmap = 1074 ms, explicit = 300 ms
Projected tok/s gain: from ~1.7 to ~2.3 (causal) or ~2.8 (with overlap)
