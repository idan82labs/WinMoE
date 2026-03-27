"""
Component 5: Slab-Based Inference Runtime

Integration approach: use llama-cpp-python for compute, but pre-load expert
weights from our aligned slab file using explicit unbuffered I/O BEFORE
llama.cpp touches them via mmap. This warms the OS page cache with our
fast explicit reads instead of letting mmap do slow page faults.

The professor said supplementing mmap degrades performance (NVMe contention).
BUT: if we do the explicit reads SEQUENTIALLY (read all K experts, THEN
let llama.cpp compute), there's no concurrent NVMe access.

This is a bridge approach:
- Proves the slab format works end-to-end
- Measures real tok/s improvement
- Doesn't require modifying llama.cpp source

Full custom engine (no mmap at all) is the final target.
"""

import ctypes
import json
import mmap
import os
import struct
import sys
import time
import threading

# Windows API for unbuffered reads
kernel32 = ctypes.windll.kernel32

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

SLAB_PATH = "D:/flash-moe-engine/experts.slab"
INDEX_PATH = "D:/flash-moe-engine/expert_index.json"

GGUF_PATH = "D:/hf_cache/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_XXS/Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf"
LLAMA_CLI = "D:/llama-cpp-src/build/bin/Release/llama-cli.exe"


class SlabReader:
    """Read expert blocks from aligned slab file using unbuffered I/O."""

    def __init__(self, slab_path, index_path):
        # Load index
        with open(index_path) as f:
            self.index = json.load(f)

        self.slot_size = self.index["slot_size"]
        self.alignment = self.index["alignment"]
        self.num_layers = self.index["num_layers"]
        self.experts_per_layer = self.index["experts_per_layer"]

        # Open slab with unbuffered I/O
        self.handle = kernel32.CreateFileW(
            slab_path,
            GENERIC_READ,
            FILE_SHARE_READ,
            None,
            OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
            None
        )
        if self.handle == INVALID_HANDLE_VALUE:
            raise OSError(f"Failed to open slab: {slab_path}")

        # Aligned read buffer
        buf_size = ((self.slot_size + self.alignment - 1) // self.alignment) * self.alignment
        self.buf = ctypes.create_string_buffer(buf_size)
        self.buf_size = buf_size

        # Stats
        self.total_reads = 0
        self.total_bytes = 0
        self.total_time_ms = 0

    def read_expert(self, layer, expert_id):
        """Read one expert block from slab. Returns bytes read."""
        layer_key = f"layer_{layer:02d}"
        expert_info = self.index["layers"][layer_key]["experts"][str(expert_id)]
        offset = expert_info["slab_offset"]

        # Align offset down
        aligned_offset = (offset // self.alignment) * self.alignment

        # Seek
        li = ctypes.c_longlong(aligned_offset)
        kernel32.SetFilePointerEx(self.handle, li, None, 0)

        # Read
        bytes_read = ctypes.c_ulong(0)
        t0 = time.perf_counter()
        kernel32.ReadFile(self.handle, self.buf, self.buf_size,
                         ctypes.byref(bytes_read), None)
        dt = (time.perf_counter() - t0) * 1000

        self.total_reads += 1
        self.total_bytes += self.slot_size
        self.total_time_ms += dt

        return self.slot_size

    def read_experts_for_layer(self, layer, expert_ids):
        """Read K experts for a layer. Sequential — no NVMe contention."""
        for eid in expert_ids:
            self.read_expert(layer, eid)

    def get_stats(self):
        if self.total_reads == 0:
            return {}
        return {
            "total_reads": self.total_reads,
            "total_MB": self.total_bytes / (1024 * 1024),
            "total_time_ms": self.total_time_ms,
            "bandwidth_MBps": (self.total_bytes / (1024 * 1024)) / (self.total_time_ms / 1000) if self.total_time_ms > 0 else 0,
            "per_expert_ms": self.total_time_ms / self.total_reads,
        }

    def close(self):
        kernel32.CloseHandle(self.handle)


def warmup_experts_from_slab(slab_reader, num_layers=60, k=3, seed=42):
    """
    Pre-warm the OS page cache by reading likely expert blocks from the slab
    using fast unbuffered I/O. This ensures when llama.cpp's mmap touches
    these pages, they're already resident — no slow page faults.

    Uses a simple frequency-based prediction: read the statically hottest
    experts per layer.
    """
    import random
    random.seed(seed)

    t0 = time.perf_counter()
    for layer in range(num_layers):
        # Read K random experts (in production: use static hotset)
        expert_ids = random.sample(range(512), k)
        slab_reader.read_experts_for_layer(layer, expert_ids)

    dt = (time.perf_counter() - t0) * 1000
    stats = slab_reader.get_stats()
    print(f"Pre-warmed {stats['total_reads']} experts in {dt:.0f} ms "
          f"({stats['bandwidth_MBps']:.0f} MB/s)")
    return stats


def benchmark_slab_vs_baseline():
    """
    Compare:
    1. Baseline: llama-cli with cold/warm mmap
    2. Slab pre-warm: read experts from slab, then run llama-cli
    """
    import subprocess

    prompt = "What is quantum computing?"
    n_predict = 50

    cmd = [
        LLAMA_CLI,
        "--model", GGUF_PATH,
        "--n-gpu-layers", "0",
        "--ctx-size", "512",
        "--n-predict", str(n_predict),
        "--threads", "12",
        "--prompt", prompt,
        "--no-warmup",
        "--no-display-prompt",
        "--single-turn",
        "--reasoning", "off",
    ]

    print("=" * 70)
    print("Slab Inference Benchmark — Component 5")
    print("=" * 70)

    # Test 1: Baseline (mmap, whatever cache state)
    print("\n--- Test 1: Baseline (mmap) ---")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                           encoding='utf-8', errors='replace')
    dt = time.perf_counter() - t0
    parse_llama_output(result.stdout + "\n" + result.stderr, "Baseline", dt)

    # Test 2: Slab pre-warm, then inference
    print("\n--- Test 2: Slab pre-warm + inference ---")
    slab = SlabReader(SLAB_PATH, INDEX_PATH)
    warmup_stats = warmup_experts_from_slab(slab, num_layers=60, k=3)
    slab.close()

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                           encoding='utf-8', errors='replace')
    dt = time.perf_counter() - t0
    parse_llama_output(result.stdout + "\n" + result.stderr, "Slab pre-warm", dt)

    # Test 3: Second run (fully warm)
    print("\n--- Test 3: Fully warm (second run) ---")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                           encoding='utf-8', errors='replace')
    dt = time.perf_counter() - t0
    parse_llama_output(result.stdout + "\n" + result.stderr, "Fully warm", dt)

    print("\n" + "=" * 70)
    print("Slab reader stats:", json.dumps(warmup_stats, indent=2))
    print("=" * 70)


def parse_llama_output(output, label, wall_time):
    """Parse llama.cpp perf output — handles both old and new formats."""
    import re
    eval_tps = 0
    prompt_tps = 0

    # New format: [ Prompt: 1.2 t/s | Generation: 1.3 t/s ]
    m = re.search(r'Generation:\s*([\d.]+)\s*t/s', output)
    if m: eval_tps = float(m.group(1))
    m = re.search(r'Prompt:\s*([\d.]+)\s*t/s', output)
    if m: prompt_tps = float(m.group(1))

    # Old format: X.XX tokens per second
    if eval_tps == 0:
        for line in output.split("\n"):
            if "eval time" in line and "prompt" not in line:
                m2 = re.search(r'([\d.]+) tokens per second', line)
                if m2: eval_tps = float(m2.group(1))
            if "prompt eval time" in line:
                m2 = re.search(r'([\d.]+) tokens per second', line)
                if m2: prompt_tps = float(m2.group(1))

    print(f"  {label}: gen={eval_tps:.2f} tok/s, prompt={prompt_tps:.2f} tok/s, "
          f"wall={wall_time:.1f}s")
    return eval_tps, prompt_tps


if __name__ == "__main__":
    if "--warmup-only" in sys.argv:
        slab = SlabReader(SLAB_PATH, INDEX_PATH)
        warmup_experts_from_slab(slab)
        slab.close()
    else:
        benchmark_slab_vs_baseline()
