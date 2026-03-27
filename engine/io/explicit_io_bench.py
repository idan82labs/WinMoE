"""
Explicit I/O Benchmark — Workstream 3

Compares mmap vs explicit offset-based reads for expert-sized blocks.
Uses the actual GGUF shard files to measure real-world performance.

Tests:
  1. mmap sequential (current llama.cpp approach)
  2. mmap random (simulates cache-miss expert access)
  3. Explicit unbuffered sequential (FILE_FLAG_NO_BUFFERING)
  4. Explicit unbuffered random
  5. Explicit buffered random (with OS cache)
"""
import ctypes
import ctypes.wintypes
import mmap
import os
import random
import struct
import sys
import time

# Expert block size at IQ2_XXS
EXPERT_SIZE = 3 * 1024 * 1024 + 512 * 1024  # ~3.5 MB (rounded to 4KB alignment)
EXPERT_SIZE_ALIGNED = ((EXPERT_SIZE + 4095) // 4096) * 4096  # 4KB aligned
K_VALUES = [3, 5, 8]
NUM_TRIALS = 20  # number of layer-loads to simulate per test

SHARD = "D:/hf_cache/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_XXS/Qwen3.5-397B-A17B-UD-IQ2_XXS-00002-of-00004.gguf"

# Win32 API
kernel32 = ctypes.windll.kernel32

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 1
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_OVERLAPPED = 0x40000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


def bench_mmap_random(path, num_reads, read_size):
    """Read random expert-sized chunks via mmap."""
    file_size = os.path.getsize(path)
    max_offset = file_size - read_size
    offsets = [random.randint(0, max_offset) for _ in range(num_reads)]

    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        t0 = time.perf_counter()
        total_bytes = 0
        for off in offsets:
            _ = mm[off:off + read_size]
            total_bytes += read_size
        elapsed = time.perf_counter() - t0
        mm.close()

    return total_bytes, elapsed


def bench_explicit_unbuffered(path, num_reads, read_size):
    """Read random expert-sized chunks via explicit unbuffered I/O."""
    file_size = os.path.getsize(path)
    aligned_size = ((read_size + 4095) // 4096) * 4096
    max_offset = (file_size - aligned_size) // 4096 * 4096
    offsets = [(random.randint(0, max_offset) // 4096) * 4096 for _ in range(num_reads)]

    # Open with NO_BUFFERING
    handle = kernel32.CreateFileW(
        path, GENERIC_READ, FILE_SHARE_READ, None,
        OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, None
    )
    if handle == INVALID_HANDLE_VALUE:
        return 0, 0

    # Aligned buffer
    buf = ctypes.create_string_buffer(aligned_size)
    bytes_read = ctypes.wintypes.DWORD()

    t0 = time.perf_counter()
    total_bytes = 0
    for off in offsets:
        # Seek
        high = ctypes.wintypes.LONG(off >> 32)
        kernel32.SetFilePointer(handle, ctypes.wintypes.LONG(off & 0xFFFFFFFF), ctypes.byref(high), 0)
        # Read
        ok = kernel32.ReadFile(handle, buf, aligned_size, ctypes.byref(bytes_read), None)
        if ok:
            total_bytes += bytes_read.value
    elapsed = time.perf_counter() - t0

    kernel32.CloseHandle(handle)
    return total_bytes, elapsed


def bench_explicit_buffered(path, num_reads, read_size):
    """Read random expert-sized chunks via standard buffered I/O."""
    file_size = os.path.getsize(path)
    max_offset = file_size - read_size
    offsets = [random.randint(0, max_offset) for _ in range(num_reads)]

    with open(path, 'rb') as f:
        t0 = time.perf_counter()
        total_bytes = 0
        for off in offsets:
            f.seek(off)
            data = f.read(read_size)
            total_bytes += len(data)
        elapsed = time.perf_counter() - t0

    return total_bytes, elapsed


def main():
    if not os.path.exists(SHARD):
        print(f"ERROR: Shard not found: {SHARD}")
        sys.exit(1)

    file_size = os.path.getsize(SHARD)
    print("=" * 70)
    print("Explicit I/O Benchmark — Expert-Sized Reads")
    print("=" * 70)
    print(f"File: {os.path.basename(SHARD)} ({file_size / 1e9:.1f} GB)")
    print(f"Expert block: {EXPERT_SIZE / 1e6:.1f} MB (aligned: {EXPERT_SIZE_ALIGNED / 1e6:.1f} MB)")
    print()

    for K in K_VALUES:
        num_reads = K * NUM_TRIALS  # K experts per layer × N layers
        read_size = EXPERT_SIZE_ALIGNED

        print(f"--- K={K}: {num_reads} reads of {read_size / 1e6:.1f} MB ---")

        # mmap random
        total, elapsed = bench_mmap_random(SHARD, num_reads, read_size)
        bw = total / elapsed / 1e6 if elapsed > 0 else 0
        per_expert = elapsed / num_reads * 1000
        print(f"  mmap random:       {bw:>7.0f} MB/s  ({per_expert:.2f} ms/expert)")

        # explicit unbuffered random
        total, elapsed = bench_explicit_unbuffered(SHARD, num_reads, read_size)
        bw = total / elapsed / 1e6 if elapsed > 0 else 0
        per_expert = elapsed / num_reads * 1000
        print(f"  explicit unbuf:    {bw:>7.0f} MB/s  ({per_expert:.2f} ms/expert)")

        # explicit buffered random
        total, elapsed = bench_explicit_buffered(SHARD, num_reads, read_size)
        bw = total / elapsed / 1e6 if elapsed > 0 else 0
        per_expert = elapsed / num_reads * 1000
        print(f"  explicit buffered: {bw:>7.0f} MB/s  ({per_expert:.2f} ms/expert)")

        print()


if __name__ == "__main__":
    main()
