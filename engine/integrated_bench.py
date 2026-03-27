"""
Integrated Expert Service Engine Benchmark

Simulates the full expert-byte service path with REAL disk I/O:
  SSD (explicit unbuffered) -> pinned RAM staging -> GPU (cudaMemcpyAsync)

Uses the actual GGUF shard files as the data source. Reads expert-sized blocks
at random offsets (simulating cache misses) and stages them through pinned memory
to the GPU.

This measures the end-to-end integrated path, not individual components.

Comparison targets:
  - mmap baseline: 586 MB/s, ~5.96 ms/expert
  - explicit unbuf: 2200 MB/s, ~1.67 ms/expert (standalone)
  - integrated target: prove the full SSD->RAM->GPU pipeline is >2x faster than mmap
"""
import ctypes
import ctypes.wintypes
import os
import sys
import time
import random
import json
import struct
from concurrent.futures import ThreadPoolExecutor

# Try to import torch for GPU staging
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

# Win32 API
kernel32 = ctypes.windll.kernel32
GENERIC_READ = 0x80000000
FILE_SHARE_READ = 1
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_OVERLAPPED = 0x40000000
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

# Model parameters
EXPERT_SIZE = 3495253  # 3.5 MB per expert (IQ2_XXS)
EXPERT_SIZE_ALIGNED = ((EXPERT_SIZE + 4095) // 4096) * 4096
NUM_LAYERS = 60
EXPERTS_PER_LAYER = 512
K = 3

SHARDS = [
    "D:/hf_cache/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_XXS/Qwen3.5-397B-A17B-UD-IQ2_XXS-00002-of-00004.gguf",
    "D:/hf_cache/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_XXS/Qwen3.5-397B-A17B-UD-IQ2_XXS-00003-of-00004.gguf",
]


class ExplicitReader:
    """Read expert blocks via Win32 explicit unbuffered I/O."""

    def __init__(self, path):
        self.path = path
        self.file_size = os.path.getsize(path)
        self.handle = kernel32.CreateFileW(
            path, GENERIC_READ, FILE_SHARE_READ, None,
            OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, None
        )
        if self.handle == INVALID_HANDLE_VALUE:
            raise OSError(f"Failed to open {path}")

    def read_at(self, offset, size, buf):
        """Read `size` bytes at `offset` into `buf`."""
        aligned_offset = (offset // 4096) * 4096
        aligned_size = ((size + 4095) // 4096) * 4096
        high = ctypes.wintypes.LONG(aligned_offset >> 32)
        kernel32.SetFilePointer(
            self.handle,
            ctypes.wintypes.LONG(aligned_offset & 0xFFFFFFFF),
            ctypes.byref(high), 0
        )
        bytes_read = ctypes.wintypes.DWORD()
        ok = kernel32.ReadFile(self.handle, buf, aligned_size, ctypes.byref(bytes_read), None)
        return ok and bytes_read.value > 0

    def close(self):
        kernel32.CloseHandle(self.handle)


class MmapReader:
    """Read expert blocks via mmap (simulates llama.cpp)."""

    def __init__(self, path):
        import mmap as mmap_mod
        self.path = path
        self.file_size = os.path.getsize(path)
        self.fh = open(path, 'rb')
        self.mm = mmap_mod.mmap(self.fh.fileno(), 0, access=mmap_mod.ACCESS_READ)

    def read_at(self, offset, size, buf=None):
        """Read via mmap (copies into internal buffer)."""
        data = self.mm[offset:offset + size]
        return len(data) == size

    def close(self):
        self.mm.close()
        self.fh.close()


def generate_access_pattern(num_tokens, num_layers, K, experts_per_layer, file_size):
    """Generate random expert access offsets simulating inference."""
    pattern = []
    for t in range(num_tokens):
        token_accesses = []
        for l in range(num_layers):
            experts = random.sample(range(experts_per_layer), K)
            for e in experts:
                # Map expert to a file offset (simulated — real slab would be indexed)
                offset = (l * experts_per_layer + e) * EXPERT_SIZE_ALIGNED
                offset = offset % (file_size - EXPERT_SIZE_ALIGNED)
                offset = (offset // 4096) * 4096  # align
                token_accesses.append(offset)
        pattern.append(token_accesses)
    return pattern


def bench_integrated_mmap(pattern, shard_path):
    """Benchmark: mmap reads -> (optional GPU copy)."""
    reader = MmapReader(shard_path)
    num_tokens = len(pattern)

    token_times = []
    for t in range(num_tokens):
        t0 = time.perf_counter()
        for offset in pattern[t]:
            reader.read_at(offset, EXPERT_SIZE)
        token_times.append(time.perf_counter() - t0)

    reader.close()

    total_bytes = sum(len(p) for p in pattern) * EXPERT_SIZE
    total_time = sum(token_times)
    return {
        "method": "mmap",
        "total_bytes_MB": total_bytes / 1e6,
        "total_time_ms": total_time * 1000,
        "mean_token_ms": sum(token_times) / len(token_times) * 1000,
        "bandwidth_MBps": total_bytes / total_time / 1e6,
        "tok_s": 1000.0 / (sum(token_times) / len(token_times) * 1000),
    }


def bench_integrated_explicit(pattern, shard_path):
    """Benchmark: explicit unbuffered reads -> staging buffer."""
    reader = ExplicitReader(shard_path)
    buf = ctypes.create_string_buffer(EXPERT_SIZE_ALIGNED)
    num_tokens = len(pattern)

    token_times = []
    for t in range(num_tokens):
        t0 = time.perf_counter()
        for offset in pattern[t]:
            reader.read_at(offset, EXPERT_SIZE, buf)
        token_times.append(time.perf_counter() - t0)

    reader.close()

    total_bytes = sum(len(p) for p in pattern) * EXPERT_SIZE
    total_time = sum(token_times)
    return {
        "method": "explicit_unbuffered",
        "total_bytes_MB": total_bytes / 1e6,
        "total_time_ms": total_time * 1000,
        "mean_token_ms": sum(token_times) / len(token_times) * 1000,
        "bandwidth_MBps": total_bytes / total_time / 1e6,
        "tok_s": 1000.0 / (sum(token_times) / len(token_times) * 1000),
    }


def bench_integrated_explicit_gpu(pattern, shard_path):
    """Benchmark: explicit unbuffered reads -> pinned RAM -> GPU."""
    if not HAS_CUDA:
        return {"method": "explicit+gpu", "error": "CUDA not available"}

    reader = ExplicitReader(shard_path)

    # Pinned host buffer
    pinned_buf = torch.empty(EXPERT_SIZE_ALIGNED, dtype=torch.uint8,
                            device='cpu').pin_memory()
    pinned_ptr = pinned_buf.data_ptr()
    ctypes_buf = (ctypes.c_char * EXPERT_SIZE_ALIGNED).from_address(pinned_ptr)

    # GPU destination
    gpu_buf = torch.empty(EXPERT_SIZE_ALIGNED, dtype=torch.uint8, device='cuda:0')
    stream = torch.cuda.Stream()

    num_tokens = len(pattern)
    token_times = []

    for t in range(num_tokens):
        t0 = time.perf_counter()
        for offset in pattern[t]:
            # SSD -> pinned RAM
            reader.read_at(offset, EXPERT_SIZE, ctypes_buf)
            # Pinned RAM -> GPU (async)
            with torch.cuda.stream(stream):
                gpu_buf.copy_(pinned_buf, non_blocking=True)
        stream.synchronize()
        token_times.append(time.perf_counter() - t0)

    reader.close()

    total_bytes = sum(len(p) for p in pattern) * EXPERT_SIZE
    total_time = sum(token_times)
    return {
        "method": "explicit+pinned+gpu",
        "total_bytes_MB": total_bytes / 1e6,
        "total_time_ms": total_time * 1000,
        "mean_token_ms": sum(token_times) / len(token_times) * 1000,
        "bandwidth_MBps": total_bytes / total_time / 1e6,
        "tok_s": 1000.0 / (sum(token_times) / len(token_times) * 1000),
    }


def bench_integrated_explicit_gpu_pipelined(pattern, shard_path):
    """Benchmark: pipelined double-buffer — read expert N+1 while GPU processes expert N."""
    if not HAS_CUDA:
        return {"method": "pipelined+gpu", "error": "CUDA not available"}

    reader = ExplicitReader(shard_path)

    # Double-buffered pinned host
    pinned_a = torch.empty(EXPERT_SIZE_ALIGNED, dtype=torch.uint8, device='cpu').pin_memory()
    pinned_b = torch.empty(EXPERT_SIZE_ALIGNED, dtype=torch.uint8, device='cpu').pin_memory()
    ptr_a = (ctypes.c_char * EXPERT_SIZE_ALIGNED).from_address(pinned_a.data_ptr())
    ptr_b = (ctypes.c_char * EXPERT_SIZE_ALIGNED).from_address(pinned_b.data_ptr())
    bufs = [(pinned_a, ptr_a), (pinned_b, ptr_b)]

    gpu_buf = torch.empty(EXPERT_SIZE_ALIGNED, dtype=torch.uint8, device='cuda:0')
    stream = torch.cuda.Stream()

    num_tokens = len(pattern)
    token_times = []

    for t in range(num_tokens):
        accesses = pattern[t]
        t0 = time.perf_counter()

        # Pipeline: overlap SSD read of expert[i+1] with GPU transfer of expert[i]
        for i, offset in enumerate(accesses):
            buf_idx = i % 2
            pinned, ptr = bufs[buf_idx]

            # SSD -> pinned RAM
            reader.read_at(offset, EXPERT_SIZE, ptr)

            # Pinned RAM -> GPU (async, overlaps with next SSD read)
            with torch.cuda.stream(stream):
                gpu_buf.copy_(pinned, non_blocking=True)

        stream.synchronize()
        token_times.append(time.perf_counter() - t0)

    reader.close()

    total_bytes = sum(len(p) for p in pattern) * EXPERT_SIZE
    total_time = sum(token_times)
    return {
        "method": "pipelined_double_buffer+gpu",
        "total_bytes_MB": total_bytes / 1e6,
        "total_time_ms": total_time * 1000,
        "mean_token_ms": sum(token_times) / len(token_times) * 1000,
        "bandwidth_MBps": total_bytes / total_time / 1e6,
        "tok_s": 1000.0 / (sum(token_times) / len(token_times) * 1000),
    }


def main():
    shard = None
    for s in SHARDS:
        if os.path.exists(s):
            shard = s
            break
    if not shard:
        print("ERROR: No shard file found")
        sys.exit(1)

    file_size = os.path.getsize(shard)

    print("=" * 70)
    print("Integrated Expert Service Engine Benchmark")
    print("=" * 70)
    print(f"Shard: {os.path.basename(shard)} ({file_size / 1e9:.1f} GB)")
    print(f"Expert block: {EXPERT_SIZE / 1e6:.1f} MB (aligned: {EXPERT_SIZE_ALIGNED / 1e6:.1f} MB)")
    print(f"K={K}, {NUM_LAYERS} layers, {K * NUM_LAYERS} = {K * NUM_LAYERS} reads/token")
    print(f"Expert bytes per token: {K * NUM_LAYERS * EXPERT_SIZE / 1e6:.0f} MB")
    print(f"CUDA available: {HAS_CUDA}")

    NUM_TOKENS = 10  # simulate 10 tokens of full inference I/O
    print(f"\nSimulating {NUM_TOKENS} tokens of expert I/O...")
    pattern = generate_access_pattern(NUM_TOKENS, NUM_LAYERS, K,
                                      EXPERTS_PER_LAYER, file_size)

    # Compute reference: 7.12 ms/layer × 60 layers = 427 ms compute
    COMPUTE_MS = 7.12 * NUM_LAYERS

    results = []

    # 1. mmap baseline
    print("\n--- mmap (llama.cpp baseline) ---")
    r = bench_integrated_mmap(pattern, shard)
    io_ms = r["mean_token_ms"]
    causal_ms = io_ms + COMPUTE_MS
    overlap_ms = max(io_ms, COMPUTE_MS)
    r["projected_causal_tps"] = 1000 / causal_ms
    r["projected_overlap_tps"] = 1000 / overlap_ms
    print(f"  I/O only: {io_ms:.0f} ms/token ({r['bandwidth_MBps']:.0f} MB/s)")
    print(f"  + compute ({COMPUTE_MS:.0f} ms): causal={causal_ms:.0f} ms ({r['projected_causal_tps']:.2f} tok/s), overlap={overlap_ms:.0f} ms ({r['projected_overlap_tps']:.2f} tok/s)")
    results.append(r)

    # 2. explicit unbuffered
    print("\n--- explicit unbuffered (custom engine) ---")
    r = bench_integrated_explicit(pattern, shard)
    io_ms = r["mean_token_ms"]
    causal_ms = io_ms + COMPUTE_MS
    overlap_ms = max(io_ms, COMPUTE_MS)
    r["projected_causal_tps"] = 1000 / causal_ms
    r["projected_overlap_tps"] = 1000 / overlap_ms
    print(f"  I/O only: {io_ms:.0f} ms/token ({r['bandwidth_MBps']:.0f} MB/s)")
    print(f"  + compute ({COMPUTE_MS:.0f} ms): causal={causal_ms:.0f} ms ({r['projected_causal_tps']:.2f} tok/s), overlap={overlap_ms:.0f} ms ({r['projected_overlap_tps']:.2f} tok/s)")
    results.append(r)

    # 3. explicit + pinned GPU
    if HAS_CUDA:
        print("\n--- explicit + pinned RAM + GPU (full pipeline) ---")
        r = bench_integrated_explicit_gpu(pattern, shard)
        if "error" not in r:
            io_ms = r["mean_token_ms"]
            causal_ms = io_ms + COMPUTE_MS
            overlap_ms = max(io_ms, COMPUTE_MS)
            r["projected_causal_tps"] = 1000 / causal_ms
            r["projected_overlap_tps"] = 1000 / overlap_ms
            print(f"  I/O+staging: {io_ms:.0f} ms/token ({r['bandwidth_MBps']:.0f} MB/s)")
            print(f"  + compute ({COMPUTE_MS:.0f} ms): causal={causal_ms:.0f} ms ({r['projected_causal_tps']:.2f} tok/s), overlap={overlap_ms:.0f} ms ({r['projected_overlap_tps']:.2f} tok/s)")
            results.append(r)

        # 4. pipelined double-buffer
        print("\n--- pipelined double-buffer + GPU (overlap SSD & GPU transfer) ---")
        r = bench_integrated_explicit_gpu_pipelined(pattern, shard)
        if "error" not in r:
            io_ms = r["mean_token_ms"]
            causal_ms = io_ms + COMPUTE_MS
            overlap_ms = max(io_ms, COMPUTE_MS)
            r["projected_causal_tps"] = 1000 / causal_ms
            r["projected_overlap_tps"] = 1000 / overlap_ms
            print(f"  I/O+staging: {io_ms:.0f} ms/token ({r['bandwidth_MBps']:.0f} MB/s)")
            print(f"  + compute ({COMPUTE_MS:.0f} ms): causal={causal_ms:.0f} ms ({r['projected_causal_tps']:.2f} tok/s), overlap={overlap_ms:.0f} ms ({r['projected_overlap_tps']:.2f} tok/s)")
            results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<35} {'I/O ms':>8} {'Causal tps':>11} {'Overlap tps':>12}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            continue
        print(f"{r['method']:<35} {r['mean_token_ms']:>7.0f} {r.get('projected_causal_tps',0):>10.2f} {r.get('projected_overlap_tps',0):>11.2f}")

    # Save
    outpath = os.path.join(os.path.dirname(__file__), "benchmarks", "integrated_results.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
