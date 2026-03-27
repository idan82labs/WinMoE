"""
Stage 0: Measure current llama.cpp expert-byte service path.

Estimates the per-layer timing breakdown by measuring:
1. Total token time (from llama.cpp perf output)
2. Theoretical compute time (from PCIe benchmark data)
3. Expert I/O time (total - compute = I/O)
4. SSD bandwidth utilization (expert bytes / I/O time)

Also measures mmap page fault behavior for expert weight access.
"""
import os
import re
import subprocess
import time
import json
import mmap
import struct

GGUF = "D:/hf_cache/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_XXS/Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf"
LLAMA_CLI = "D:/llama-cpp-cuda/bin/llama-cli.exe"

# Model parameters (IQ2_XXS)
NUM_LAYERS = 60
EXPERTS_PER_LAYER = 512
K = 3  # active experts
TOTAL_MODEL_SIZE = 107 * 1024**3  # ~107 GB
SHARED_WEIGHT_SIZE = 7 * 1024**3  # ~7 GB (attention, embeddings, norms, routing)
EXPERT_TOTAL_SIZE = TOTAL_MODEL_SIZE - SHARED_WEIGHT_SIZE  # ~100 GB
EXPERT_BLOCK_SIZE = EXPERT_TOTAL_SIZE // (NUM_LAYERS * EXPERTS_PER_LAYER)  # per expert

# Calibrated timing constants (from Phase 0)
SSD_BANDWIDTH = 2.1e9  # 2.1 GB/s (Samsung 980 measured)
SSD_OVERHEAD_PER_REQ = 42e-6  # 42 us per request
PCIE_BANDWIDTH = 19e9  # 19 GB/s pinned
PCIE_OVERHEAD_PER_REQ = 11e-6  # 11 us per request


def run_llama(n_predict=50, n_gpu_layers=8, threads=12, prompt="What is quantum computing?"):
    """Run llama-cli and parse timing."""
    cmd = [
        LLAMA_CLI, "--model", GGUF,
        "--n-gpu-layers", str(n_gpu_layers),
        "--ctx-size", "512",
        "--n-predict", str(n_predict),
        "--threads", str(threads),
        "--prompt", prompt,
        "--no-warmup", "--no-display-prompt",
        "--single-turn", "--reasoning", "off",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                          encoding='utf-8', errors='replace')
    output = result.stderr + "\n" + result.stdout

    eval_tps = 0
    eval_ms = 0
    prompt_tps = 0
    for line in output.split("\n"):
        # New format: [ Prompt: X t/s | Generation: Y t/s ]
        m = re.search(r'Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s', line)
        if m:
            prompt_tps = float(m.group(1))
            eval_tps = float(m.group(2))
            eval_ms = 1000.0 / eval_tps if eval_tps > 0 else 0
        # Old format fallback
        if "eval time" in line and "prompt" not in line:
            m2 = re.search(r'([\d.]+) ms per token.*?([\d.]+) tokens per second', line)
            if m2:
                eval_ms = float(m2.group(1))
                eval_tps = float(m2.group(2))
        elif "prompt eval time" in line:
            m2 = re.search(r'([\d.]+) tokens per second', line)
            if m2:
                prompt_tps = float(m2.group(1))

    return {"eval_ms_per_token": eval_ms, "eval_tps": eval_tps, "prompt_tps": prompt_tps}


def estimate_breakdown(ms_per_token):
    """Estimate per-layer timing breakdown."""
    ms_per_layer = ms_per_token / NUM_LAYERS

    # Expert I/O per layer (K experts read from SSD)
    expert_bytes_per_layer = K * EXPERT_BLOCK_SIZE
    ssd_read_ms = (expert_bytes_per_layer / SSD_BANDWIDTH + K * SSD_OVERHEAD_PER_REQ) * 1000

    # PCIe transfer per layer (expert bytes RAM->GPU)
    pcie_ms = (expert_bytes_per_layer / PCIE_BANDWIDTH + K * PCIE_OVERHEAD_PER_REQ) * 1000

    # Compute (everything else: attention, routing, expert FFN, norms)
    compute_ms = ms_per_layer - ssd_read_ms - pcie_ms
    if compute_ms < 0:
        compute_ms = 0  # model estimate may be off

    return {
        "ms_per_layer": round(ms_per_layer, 2),
        "ssd_read_ms": round(ssd_read_ms, 2),
        "pcie_transfer_ms": round(pcie_ms, 2),
        "compute_ms": round(compute_ms, 2),
        "expert_bytes_per_layer": expert_bytes_per_layer,
        "expert_bytes_per_token": expert_bytes_per_layer * NUM_LAYERS,
        "ssd_utilization_pct": round(ssd_read_ms / ms_per_layer * 100, 1),
    }


def measure_mmap_behavior():
    """Measure mmap page fault latency on expert weight file."""
    # Read a random 3.3MB chunk to simulate expert access
    shard2 = GGUF.replace("00001-of-00004", "00002-of-00004")
    if not os.path.exists(shard2):
        return {"error": "shard2 not found"}

    file_size = os.path.getsize(shard2)
    offsets = [i * EXPERT_BLOCK_SIZE for i in range(0, min(100, file_size // EXPERT_BLOCK_SIZE))]

    # Sequential read via mmap (simulates warm cache)
    t0 = time.perf_counter()
    with open(shard2, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        bytes_read = 0
        for off in offsets[:50]:
            if off + EXPERT_BLOCK_SIZE <= file_size:
                _ = mm[off:off + EXPERT_BLOCK_SIZE]
                bytes_read += EXPERT_BLOCK_SIZE
        mm.close()
    mmap_time = time.perf_counter() - t0

    return {
        "mmap_50_experts_ms": round(mmap_time * 1000, 1),
        "mmap_bytes_read": bytes_read,
        "mmap_bandwidth_MBps": round(bytes_read / mmap_time / 1e6, 0) if mmap_time > 0 else 0,
    }


def main():
    print("=" * 70)
    print("Stage 0: Current Expert-Byte Service Path Measurement")
    print("=" * 70)

    print(f"\nModel: Qwen3.5-397B-A17B IQ2_XXS")
    print(f"Expert block size: {EXPERT_BLOCK_SIZE / 1e6:.2f} MB")
    print(f"Expert bytes per layer (K={K}): {K * EXPERT_BLOCK_SIZE / 1e6:.2f} MB")
    print(f"Expert bytes per token: {K * EXPERT_BLOCK_SIZE * NUM_LAYERS / 1e6:.2f} MB")

    # Run warm inference
    print(f"\n--- Warm inference (K={K}, ngl=8, t=12) ---")
    result = run_llama(n_predict=100)
    print(f"  eval: {result['eval_ms_per_token']:.1f} ms/token ({result['eval_tps']:.2f} tok/s)")
    print(f"  prompt: {result['prompt_tps']:.2f} tok/s")

    # Breakdown
    breakdown = estimate_breakdown(result['eval_ms_per_token'])
    print(f"\n--- Estimated per-layer breakdown ---")
    print(f"  Total per layer:    {breakdown['ms_per_layer']:.2f} ms")
    print(f"  SSD read:           {breakdown['ssd_read_ms']:.2f} ms ({breakdown['ssd_utilization_pct']}%)")
    print(f"  PCIe transfer:      {breakdown['pcie_transfer_ms']:.2f} ms")
    print(f"  Compute:            {breakdown['compute_ms']:.2f} ms")
    print(f"  Expert bytes/token: {breakdown['expert_bytes_per_token'] / 1e6:.1f} MB")

    # mmap behavior
    print(f"\n--- mmap page fault behavior ---")
    mmap_result = measure_mmap_behavior()
    print(f"  50 experts via mmap: {mmap_result.get('mmap_50_experts_ms', 'N/A')} ms")
    print(f"  Bandwidth: {mmap_result.get('mmap_bandwidth_MBps', 'N/A')} MB/s")

    # Save results
    results = {
        "stage": 0,
        "model": "Qwen3.5-397B-A17B-IQ2_XXS",
        "K": K,
        "n_gpu_layers": 8,
        "threads": 12,
        "inference": result,
        "breakdown": breakdown,
        "mmap": mmap_result,
        "calibration": {
            "ssd_bandwidth_GBps": SSD_BANDWIDTH / 1e9,
            "pcie_bandwidth_GBps": PCIE_BANDWIDTH / 1e9,
            "expert_block_size_MB": EXPERT_BLOCK_SIZE / 1e6,
        }
    }

    outpath = os.path.join(os.path.dirname(__file__), "stage0_results.json")
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
