# Custom MoE Inference Engine — Architecture Plan

## Status: Waiting for Professor Answers (V3 Engine Questions)

Questions sent: `docs/professor_questions_v3_engine.md`
Expected turnaround: ~30-40 minutes

## Direction

Abandon llama.cpp modifications. Build a custom engine from scratch that owns
the entire SSD→RAM→GPU pipeline. No mmap. No OS page cache dependency.

## What We Know (Locked)

### Measured Data Points
- mmap expert reads: 586 MB/s (measured, `engine/io/explicit_io_bench.py`)
- Explicit unbuffered: 2,200 MB/s (measured, same benchmark)
- Pinned H2D: 19 GB/s (measured, `work/baseline/pcie_benchmark.py`)
- Compute per layer: 7.12 ms (derived from Stage 0, `engine/benchmarks/stage0_results.json`)
- Current best tok/s: 1.7 (measured, llama.cpp source-built AVX512, mmap)
- Compute ceiling: 2.34 tok/s (professor-derived, T_c = 427 ms/token)
- K reduction is free at IQ2_XXS (professor-confirmed)
- Prediction is a dead end (professor: Fano bound, 24.4-bit label entropy)
- Gen4 NVMe doesn't help batch-1 after mmap replacement (professor-confirmed)

### Architecture-Independent Components (Build Now)
These are needed regardless of which pipeline architecture the professor recommends:

1. **Slab Repacker** — extract expert weights from GGUF into aligned slab files
2. **Explicit I/O Reader** — read from slabs at 2200 MB/s, reusable component
3. **Tiered Cache Manager** — VRAM hot / RAM warm / SSD cold data structures

### Professor Will Lock
- Pipeline scheduling: causal vs overlap vs double-buffer
- Memory tier sizes: optimal (C_v, C_r) allocation
- NVMe queue isolation: can separate queues avoid contention?
- T_c ceiling: can GPU kernels reduce compute below 7.12 ms/layer?

## Component Specifications

### Component 1: Slab Repacker

**Input**: GGUF shard files (4 files, 107 GB total)
**Output**: Aligned slab files + index

Format:
```
Per-layer slab: layer_XX.slab
  [4KB aligned header: magic, version, num_experts, expert_size, alignment]
  [expert_0 data, padded to 4KB]
  [expert_1 data, padded to 4KB]
  ...
  [expert_511 data, padded to 4KB]

Index file: expert_index.json
  Per layer: file path, expert count, expert size, offset table
```

Requirements:
- 4KB alignment for FILE_FLAG_NO_BUFFERING compatibility
- Sequential layout within each layer for maximum NVMe throughput
- Support reading individual expert blocks by (layer, expert_id) lookup

### Component 2: Explicit I/O Reader

A C++ class wrapping Windows unbuffered I/O:
- Open slab file with FILE_FLAG_NO_BUFFERING
- Read expert block at known offset with sector alignment
- Async support via IOCP for pipeline overlap (if professor says possible)
- Benchmark: must achieve 2000+ MB/s on 3.5 MB reads

API:
```cpp
class ExpertReader {
    ExpertReader(const char* slab_path);
    void read_expert(int expert_id, void* dest, size_t size);  // sync
    void read_expert_async(int expert_id, void* dest, size_t size, OVERLAPPED* ov);  // async
    double get_bandwidth_mbps() const;
};
```

### Component 3: Tiered Cache Manager

Data structures for three-tier expert caching:
- VRAM tier: fixed-size array of expert blocks in GPU memory
- RAM tier: fixed-size array in pinned host memory (cudaHostAlloc)
- Lookup: (layer_id, expert_id) → {tier, offset} or MISS

API:
```cpp
class TieredCache {
    TieredCache(size_t vram_blocks, size_t ram_blocks);

    enum Tier { VRAM, RAM, SSD_MISS };

    struct LookupResult { Tier tier; void* data; };
    LookupResult lookup(int layer_id, int expert_id);

    void insert_vram(int layer_id, int expert_id, const void* data);
    void insert_ram(int layer_id, int expert_id, const void* data);
    void init_static_hotset(const std::map<std::pair<int,int>, int>& frequency);

    CacheStats get_stats() const;
};
```

## Build Order

1. Slab repacker (Python, runs once offline)
2. Explicit I/O reader (C++, standalone benchmark)
3. Tiered cache manager (C++, standalone benchmark with simulated access)
4. **WAIT FOR PROFESSOR** → wiring architecture
5. Pipeline controller (wires reader + cache + compute)
6. Compute kernels (dequant, matmul, attention — may reuse ggml)
7. Integration + end-to-end benchmark

## What to Do While Waiting

Build components 1-3 with standalone benchmarks. Each must pass its own test:
- Repacker: produces valid slab files, readable by component 2
- Reader: achieves 2000+ MB/s on slab reads
- Cache: correct hit/miss tracking, LFU eviction works

When professor answers arrive, we wire them together per the recommended architecture.
