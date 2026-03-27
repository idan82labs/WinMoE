"""
Benchmark llama.cpp CUDA binary on Qwen3.5-397B.
Runs llama-cli with various n_gpu_layers and parses output.
"""
import os
import re
import subprocess
import sys
import time

LLAMA_CLI = "D:/llama-cpp-cuda/bin/llama-cli.exe"
GGUF = "D:/hf_cache/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_XXS/Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf"
PROMPT = "What is quantum computing?"
CTX = 512
N_PREDICT = 50


def run_bench(n_gpu_layers: int, k_note: str = "5") -> dict:
    cmd = [
        LLAMA_CLI,
        "--model", GGUF,
        "--n-gpu-layers", str(n_gpu_layers),
        "--ctx-size", str(CTX),
        "--n-predict", str(N_PREDICT),
        "--prompt", PROMPT,
        "--no-warmup",
        "--no-display-prompt",
        "--no-conversation",
        "--single-turn",
        "--reasoning", "off",
    ]

    print(f"\n--- n_gpu_layers={n_gpu_layers} ---")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, encoding='utf-8', errors='replace')
    wall = time.time() - t0

    output = result.stderr + "\n" + result.stdout

    # Parse llama_perf lines
    prompt_tps = 0
    eval_tps = 0
    eval_ms = 0
    ttft = 0

    for line in output.split("\n"):
        if "prompt eval time" in line:
            m = re.search(r'([\d.]+) tokens per second', line)
            if m:
                prompt_tps = float(m.group(1))
            m2 = re.search(r'prompt eval time\s*=\s*([\d.]+) ms', line)
            if m2:
                ttft = float(m2.group(1))
        if "eval time" in line and "prompt" not in line:
            m = re.search(r'([\d.]+) tokens per second', line)
            if m:
                eval_tps = float(m.group(1))
            m2 = re.search(r'([\d.]+) ms per token', line)
            if m2:
                eval_ms = float(m2.group(1))

    # Check CUDA active
    cuda_active = "ggml_cuda_init" in output

    print(f"  CUDA: {'yes' if cuda_active else 'NO'}")
    print(f"  TTFT: {ttft:.0f} ms")
    print(f"  Decode: {eval_tps:.2f} tok/s ({eval_ms:.0f} ms/tok)")
    print(f"  Prompt: {prompt_tps:.2f} tok/s")
    print(f"  Wall: {wall:.1f}s")

    return {
        "n_gpu_layers": n_gpu_layers,
        "cuda_active": cuda_active,
        "ttft_ms": round(ttft),
        "tok_s": round(eval_tps, 2),
        "ms_per_token": round(eval_ms),
        "prompt_tps": round(prompt_tps, 2),
        "wall_s": round(wall, 1),
    }


if __name__ == "__main__":
    layers = [0, 1, 5, 10, 15, 20]
    if len(sys.argv) > 1:
        layers = [int(x) for x in sys.argv[1:]]

    results = []
    for ngl in layers:
        r = run_bench(ngl)
        results.append(r)

    print(f"\n{'='*80}")
    print(f"{'ngl':>5} {'CUDA':>6} {'TTFT':>8} {'tok/s':>8} {'ms/tok':>8} {'prompt':>8} {'wall':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['n_gpu_layers']:>5} {'yes' if r['cuda_active'] else 'no':>6} "
              f"{r['ttft_ms']:>7}ms {r['tok_s']:>7.2f} {r['ms_per_token']:>7}ms "
              f"{r['prompt_tps']:>7.2f} {r['wall_s']:>7.1f}s")
    print("=" * 80)
