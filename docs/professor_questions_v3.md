# Professor Questions V3 — Architecture-Locking Decisions

## Background Context

We are running Qwen3.5-397B-A17B (a 397 billion parameter Mixture-of-Experts model)
on a Windows laptop with:
- **GPU**: RTX 3070 Laptop, 8 GB VRAM
- **RAM**: 40 GB DDR4
- **SSD**: Samsung 980 1TB NVMe (PCIe Gen3, measured 2.1 GB/s sequential unbuffered)
- **OS**: Windows 11

The model uses IQ2_XXS quantization (~2-bit), totaling ~107 GB on disk in 4 GGUF shard files.
Expert weights make up ~100 GB (~93%) of the model. The remaining ~7 GB is shared weights
(attention, embeddings, norms, routing matrices).

Architecture: 60 MoE layers, 512 experts per layer, K=3 active experts per token per layer.
Each expert block is ~3.5 MB at IQ2_XXS quantization.

**Current performance**: 1.7 tok/s using llama.cpp with mmap (memory-mapped file I/O).
**Professor-confirmed compute ceiling**: 2.34 tok/s (T_c = 7.12 ms/layer × 60 layers = 427 ms).

### What We've Measured

| I/O Method | Bandwidth (expert-sized 3.5 MB reads) | Source |
|------------|---------------------------------------|--------|
| mmap random (cold) | 586 MB/s, 5.96 ms/expert | `engine/io/explicit_io_bench.py` |
| explicit unbuffered (`FILE_FLAG_NO_BUFFERING`) | 2,200 MB/s, 1.67 ms/expert | same benchmark |
| explicit buffered (standard `ReadFile`) | 1,184 MB/s, 3.10 ms/expert | same benchmark |

These are measured on the same Samsung 980 NVMe, same expert-sized blocks (3.5 MB), random offsets
within a 49.8 GB GGUF shard file. The 3.7x speedup of explicit unbuffered over mmap is consistent
across K=3, 5, 8 at queue depth 1 (synchronous, single-threaded reads).

### What Failed

**Attempt: Background cache warming during inference** (`engine/fast_inference.py`)
- A background thread continuously reads expert files via explicit I/O to warm the OS page cache
- WHILE llama.cpp runs inference using mmap on the same files
- **Result**: Performance DROPPED from 1.5 tok/s to 0.9 tok/s (-40%)
- **Cause**: Both I/O streams compete for the same NVMe controller/queue

This matches the professor's M/G/1-PS model from V2 answers:
```
R_A ≈ s_A / (1 - ρ_A - ρ_B)
```
Adding stream B (background reads) increases stream A's (mmap faults) response time.

**Attempt: `--no-mmap` flag in llama.cpp**
- llama.cpp reads the ENTIRE 107 GB model into RAM at startup via `fread`
- With only 40 GB RAM, this would require a 67 GB page file
- C: drive has only 12 GB free — page file won't fit
- **Result**: Not viable on this hardware

### The Current Problem

We cannot use `--no-mmap` (model too large for RAM). We proved explicit I/O is 3.7x faster
than mmap, but we cannot run it concurrently with mmap (NVMe contention). We need an
architecture that captures the explicit I/O speedup without the contention.

---

## Q1: Synchronous Pre-Read Inside the Compute Loop

### Proposed Design

llama.cpp uses `ggml_compute_forward_mul_mat_id()` to compute expert FFN layers. This function
accesses expert weight data via `src0->data + (expert_id × stride)`, where `src0->data` points
into the mmap'd region. Cold expert blocks trigger page faults at 586 MB/s.

We propose: BEFORE `mul_mat_id` touches expert memory, explicitly read the K=3 needed expert
blocks (identified by the routing indices tensor) via `ReadFile(FILE_FLAG_NO_BUFFERING)` into
a staging buffer. Then EITHER:
- (a) `memcpy` the data into the mmap'd region (so subsequent access hits warm pages), OR
- (b) swap the `src0->data` pointer to point at the staging buffer for this operation

The explicit read and the compute happen **sequentially** — never concurrently. The NVMe
handles only one request stream at a time.

### Questions

1. **Does sequential explicit-then-compute avoid the M/G/1-PS contention?**
   Since there is no concurrent NVMe access (explicit read completes fully before compute starts),
   the load factor ρ_B = 0 during compute. Does this guarantee no contention?

2. **Option (a) — memcpy into mmap'd pages**: If we write data into the mmap'd virtual address range,
   does the OS guarantee those pages stay resident for the subsequent matmul read? Or could the OS
   evict/remap them between the memcpy and the compute access?

3. **Does the OS issue background read-ahead on the mmap'd file?** If Windows speculatively
   pre-fetches pages around the fault address, those background reads could overlap with our
   explicit read even though OUR code is sequential. Does this happen for random-access patterns
   (expert weight access is random across the 100 GB mapped region)?

---

## Q2: VirtualAlloc/Decommit Overhead

### Proposed Design

Instead of mmap, use `VirtualAlloc(MEM_RESERVE)` to create a 100 GB virtual address space
for expert weights. Per layer during inference:
1. `VirtualAlloc(MEM_COMMIT)` the K=3 expert blocks (3 × 3.5 MB = 10.5 MB)
2. `ReadFile(FILE_FLAG_NO_BUFFERING)` to fill them from SSD
3. Compute the expert matmul
4. `VirtualFree(MEM_DECOMMIT)` to release physical pages

This gives perfect control: we never exceed ~10.5 MB of committed expert memory per layer,
plus whatever we cache in a separate hot tier.

### Question

**What is the expected syscall overhead per call?**

- `VirtualAlloc(MEM_COMMIT)` for a 3.5 MB region (875 pages at 4 KB granularity): ? µs
- `VirtualFree(MEM_DECOMMIT)` for the same region: ? µs

For context: the `ReadFile` for 3.5 MB takes ~1,670 µs at 2.1 GB/s. If VirtualAlloc overhead
is <100 µs (~6% of read time), this approach is viable. If >500 µs (~30%), the syscall overhead
significantly erodes the I/O speedup.

3.5 MB = 875 pages. Does the kernel's page table manipulation scale linearly with page count?
Would using 2 MB large pages (2 pages instead of 875) dramatically reduce overhead?

---

## Q3: Read Granularity — Batch vs Individual

### Data

Using the professor's calibrated affine timing model:
```
t(x, m) = α + β×m + x/B
```
Where α ≈ 0, β ≈ 42 µs/request, B ≈ 2,100 MB/s (measured on Samsung 980 with
explicit unbuffered reads; see `work/baseline/results_ssd.tsv`).

For K=3 experts of 3.5 MB each:
- 3 individual reads: 3 × (42 + 3500/2100×1000) = 3 × 1709 = 5,127 µs
- 1 batched read of 10.5 MB: 42 + 10500/2100×1000 = 5,042 µs
- Savings: 85 µs (1.7%)

### Question

**Is the batch savings larger than the affine model predicts?**

NVMe controllers have internal read-ahead and command coalescing. For truly contiguous
data (a slab file with experts packed sequentially), does the NVMe deliver higher
bandwidth than for random-offset reads of the same total size?

Our benchmark measured 2,200 MB/s for both random and sequential 3.5 MB reads. But
sequential within a single 10.5 MB read might benefit from NVMe prefetch that doesn't
activate for random offsets. Is there a measurable difference?

This determines whether building a slab repacker (to make expert reads contiguous)
provides meaningful I/O benefit, or if it's only useful for cache management.

---

## Q4: Architecture Selection

### Three Candidate Designs

All share: NVMe 2.1 GB/s, β=42 µs, K=3, 60 layers, 3.5 MB/expert, T_c=7.12 ms/layer,
40 GB RAM (~37% of 100 GB expert data cacheable).

**Design A: mmap + synchronous pre-read hook**
- Keep mmap for virtual address space
- Before each `mul_mat_id`, explicitly read K=3 expert blocks via `ReadFile(NO_BUFFERING)`
- Compute accesses warm pages (no fault) or staging buffer
- Per-layer timing (assuming 33% cache miss rate):
  - Cache hit: 0 ms read + 7.12 ms compute = 7.12 ms
  - Cache miss: K×miss × 1.67 ms read + 7.12 ms compute = 1×1.67 + 7.12 = 8.79 ms
  - Weighted average: 0.67×7.12 + 0.33×8.79 = 7.67 ms/layer → 460 ms → 2.17 tok/s

**Design B: VirtualAlloc/Decommit + tiered cache**
- No mmap. VirtualAlloc reserves 100 GB address space.
- Tiered cache: VRAM hot (top 192 experts/layer), RAM warm (LRU), SSD cold
- Per-layer: commit+read only on cache miss, compute always
- No page cache — we manage all caching explicitly
- Timing depends on VirtualAlloc overhead (Q2) and cache hit rate

**Design C: Slab files + custom ggml buffer allocator**
- Repack experts into aligned slab files (one per layer)
- Custom `ggml_backend_buffer_type` that reads from slabs on demand
- Expert tensor->data points to a managed buffer that is filled via explicit I/O
- Cleanest separation, largest code change, requires model repacking step

### Question

**Given the constraints (40 GB RAM, single NVMe, batch-1 decode), which design
minimizes realistic per-token latency?**

Key factors to weigh:
- mmap page cache is "free" caching that Design B/C must replicate
- VirtualAlloc syscall overhead may dominate at K×60 calls per token
- ggml's compute graph is single-threaded — I/O and compute cannot overlap
  without adding a background I/O thread (which hits NVMe contention)
- mmap pages may get evicted by the OS under memory pressure (40 GB RAM is tight)

---

## Q5: PrefetchVirtualMemory — Bypass or Buffered?

### Context

Windows provides `PrefetchVirtualMemory(hProcess, numEntries, entries, flags)` (Win8.1+).
This API accepts an array of address ranges and "hints" the OS to bring those pages into
the working set.

If we know the virtual addresses of the K=3 needed expert blocks in the mmap'd region
(which we do — `mapping->addr() + weight->offs + expert_id × stride`), we can call
`PrefetchVirtualMemory` for those ranges before the matmul accesses them.

### Question

**What I/O mechanism does `PrefetchVirtualMemory` use internally?**

- **(a) Direct NVMe submission**: bypasses the page fault handler, issues sector-aligned
  NVMe read commands directly. If so, effective bandwidth would approach our measured
  2,200 MB/s for explicit unbuffered reads. This would be the simplest possible fix —
  no file handle changes, no pointer swizzling, just add a `PrefetchVirtualMemory` call
  before each `mul_mat_id`.

- **(b) Soft page fault trigger**: internally touches each page to trigger the standard
  demand-paging path (buffered, page-fault-driven). If so, effective bandwidth would be
  the same as mmap page faults (~586 MB/s), and this API provides no benefit over just
  letting the compute touch the pages naturally.

- **(c) Asynchronous read-ahead**: issues reads asynchronously through the file system
  cache. If so, it might be faster than (b) but still uses buffered I/O (not
  `FILE_FLAG_NO_BUFFERING`), giving ~1,184 MB/s (our measured explicit buffered speed).

The answer determines whether `PrefetchVirtualMemory` is a viable shortcut or a dead end.

---

## Q6: NVMe Multi-Queue — Pipeline Overlap

### Context

Modern NVMe controllers expose multiple hardware submission queues (Samsung 980 supports
up to 64 queues per the NVMe spec). Windows' StorNVMe driver maps I/O from different
processes/handles to these queues.

Our failed background-warming experiment showed that concurrent I/O streams on the same
NVMe degrade each other (1.5 → 0.9 tok/s). The professor's M/G/1-PS model explains this
as queue-level contention.

### Proposed Design

Open **two separate file handles** to the GGUF shard files:
- **Handle A**: regular file handle used by mmap (demand paging for non-expert tensors
  like attention, embeddings, norms — ~7 GB, fits in RAM, mostly cache hits)
- **Handle B**: opened with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`
  (explicit async reads for expert weight blocks during inference)

### Question

**Do Windows file handles map to separate NVMe submission queues?**

- If **YES**: Handle A and Handle B submit to different hardware queues. The NVMe controller
  processes them with independent scheduling. The M/G/1 contention model doesn't apply because
  the queues are independent servers. This enables:
  ```
  Layer N:   [GPU/CPU compute using cached expert data] ───────────→
  Layer N+1: [async read of K=3 experts via Handle B]  ──────→ [compute]
  ```
  With 67% cache hit rate, only 1 expert per layer misses on average.
  Async read of 1×3.5 MB takes 1.67 ms. Compute takes 7.12 ms.
  Pipeline: max(1.67, 7.12) = 7.12 ms/layer → 427 ms → **2.34 tok/s** (compute ceiling).

- If **NO**: Both handles ultimately serialize through the same NVMe queue. The M/G/1
  contention returns, and pipeline overlap provides no benefit. We must use the sequential
  approach from Q1 instead.

**This is the single most important question for whether we can reach the 2.34 tok/s
compute ceiling on this hardware.**

---

## Priority Order

1. **Q6** (NVMe multi-queue) — determines if pipeline overlap is possible → ceiling or not
2. **Q1** (synchronous pre-read) — determines if the simple sequential approach works
3. **Q5** (PrefetchVirtualMemory) — if it bypasses faults, simplest possible fix
4. **Q4** (architecture selection) — locked by answers to Q1, Q5, Q6
5. **Q2** (VirtualAlloc overhead) — only matters if Design B is chosen
6. **Q3** (read granularity) — minor optimization, lowest priority
