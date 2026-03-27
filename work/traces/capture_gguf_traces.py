"""
Capture routing traces via llama-cpp-python with Qwen3-30B-A3B GGUF.

Strategy: Run model inference, but also intercept the hidden states at each
MoE gate by using a combined approach:
1. Download and run the GGUF Q4_K_M with llama-cpp-python for fast inference
2. The model will run at reasonable speed with CPU + partial GPU offload
3. We capture routing statistics from the model's perplexity/logprobs output
   and correlate with the gate-weight analysis

Since llama.cpp doesn't expose routing decisions directly, we use an alternative
approach: run the model, then use the output_router_logits from HF transformers
with just a SINGLE forward pass (no generation loop) on the prefill tokens.
A single forward pass is much faster than generation.
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

CACHE_DIR = "D:/hf_cache"
MODEL_ID = "Qwen/Qwen3-30B-A3B"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")

PROMPTS = [
    "Explain the theory of general relativity and its implications for modern physics. Einstein's work on spacetime curvature showed that massive objects warp the fabric of space-time itself.",
    "Write a Python function that implements the A* pathfinding algorithm with a priority queue. The function should accept a graph represented as an adjacency list and return the shortest path.",
    "What are the main causes and effects of climate change on ocean ecosystems? Rising sea temperatures affect coral reefs, marine biodiversity, and ocean circulation patterns.",
    "Solve step by step: If f(x) = 3x^2 + 2x - 5, find f'(x) and determine all critical points where f'(x) = 0.",
    "Tell me a detailed story about a scientist who discovers time travel but faces an ethical dilemma about whether to change the past.",
]


def capture_with_single_forward():
    """Run single forward passes to get router logits — MUCH faster than generation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Flash-MoE V2 -- Single Forward Pass Routing Capture")
    print("=" * 70)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit, GPU+CPU split
    print("Loading model (4-bit, GPU+CPU)...")
    t0 = time.time()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    )

    # All CPU - avoids meta tensor / device_map issues
    # 4-bit model is ~15-18 GB, fits in 40 GB RAM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},  # all on GPU — will error if too large
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    dt = time.time() - t0
    print(f"Model loaded in {dt:.0f}s")

    # Print device map summary
    if hasattr(model, 'hf_device_map'):
        devices = set(str(v) for v in model.hf_device_map.values())
        print(f"Devices used: {devices}")

    all_traces = []

    for i, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt {i+1}/{len(PROMPTS)} ({len(prompt)} chars) ---")
        print(f"  {prompt[:80]}...")

        inputs = tokenizer(prompt, return_tensors="pt")
        # Move to the model's first device
        first_device = next(model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]
        print(f"  Tokens: {seq_len}")

        # Single forward pass with router logits
        t1 = time.time()
        with torch.no_grad():
            try:
                outputs = model(**inputs, output_router_logits=True)
                dt_fwd = time.time() - t1
                print(f"  Forward pass: {dt_fwd:.1f}s ({seq_len/dt_fwd:.1f} tok/s)")

                if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
                    router_logits = outputs.router_logits
                    print(f"  Router logits: {len(router_logits)} layers")

                    # Extract top-K expert IDs from router logits
                    # Each element: (batch*seq_len, num_experts)
                    num_layers = len(router_logits)
                    K = 8  # Qwen3-30B uses top-8

                    traces = np.zeros((seq_len, num_layers, K), dtype=np.int32)
                    gate_weights = np.zeros((seq_len, num_layers, K), dtype=np.float32)

                    for l, logits in enumerate(router_logits):
                        if logits is None:
                            continue
                        logits_np = logits.float().cpu().numpy()  # (seq_len, num_experts)
                        # Apply softmax for gate weights
                        exp_logits = np.exp(logits_np - logits_np.max(axis=-1, keepdims=True))
                        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
                        # Top-K
                        top_k_idx = np.argsort(-probs, axis=-1)[:, :K]
                        for t in range(min(seq_len, logits_np.shape[0])):
                            traces[t, l] = top_k_idx[t]
                            gate_weights[t, l] = probs[t, top_k_idx[t]]

                    all_traces.append(traces)

                    # Save per-prompt
                    np.savez_compressed(
                        os.path.join(OUTPUT_DIR, f"trace_prompt_{i:03d}.npz"),
                        traces=traces, gate_weights=gate_weights, prompt_idx=i
                    )
                    print(f"  Saved: {traces.shape}")
                else:
                    print("  WARNING: No router_logits in output!")
                    print(f"  Output keys: {[k for k in outputs.keys() if outputs[k] is not None]}")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Merge all traces
    if all_traces:
        merged = np.concatenate(all_traces, axis=0)
        np.savez_compressed(os.path.join(OUTPUT_DIR, "traces_merged.npz"), traces=merged)
        print(f"\nMerged traces: {merged.shape}")

        # Run analysis
        analyze_merged_traces(merged)
    else:
        print("\nNo traces captured!")


def analyze_merged_traces(traces):
    """Analyze the merged trace data."""
    num_tokens, num_layers, K = traces.shape
    num_experts = 128

    print(f"\n{'='*70}")
    print(f"REAL ROUTING TRACE ANALYSIS")
    print(f"{'='*70}")
    print(f"Tokens: {num_tokens}, Layers: {num_layers}, K: {K}")

    per_layer_freq = defaultdict(lambda: defaultdict(int))
    global_freq = defaultdict(int)

    for t in range(num_tokens):
        for l in range(num_layers):
            for k in range(K):
                eid = int(traces[t, l, k])
                per_layer_freq[l][eid] += 1
                global_freq[(l, eid)] += 1

    # Zipf and Gini per layer
    zipf_estimates = []
    gini_list = []

    for l in range(num_layers):
        freqs = sorted(per_layer_freq[l].values(), reverse=True)
        if len(freqs) < 3:
            continue

        total = sum(freqs)
        sorted_f = sorted(freqs)
        n = len(sorted_f)
        gini = (2 * sum((i+1) * f for i, f in enumerate(sorted_f)) - (n+1) * total) / (n * total)
        gini_list.append(gini)

        ranks = np.arange(1, min(51, len(freqs)+1), dtype=np.float64)
        log_r = np.log(ranks)
        log_f = np.log(np.array(freqs[:len(ranks)], dtype=np.float64))
        if len(log_r) >= 3:
            c = np.polyfit(log_r, log_f, 1)
            zipf_estimates.append(-c[0])

    mean_zipf = np.mean(zipf_estimates) if zipf_estimates else 0
    mean_gini = np.mean(gini_list) if gini_list else 0

    print(f"\nZipf exponent: mean={mean_zipf:.4f} (min={min(zipf_estimates):.4f}, max={max(zipf_estimates):.4f})")
    print(f"Gini coefficient: mean={mean_gini:.4f}")

    # Static cache hit rates
    total_accesses = num_tokens * num_layers * K
    sorted_global = sorted(global_freq.items(), key=lambda x: -x[1])
    total_blocks = num_layers * num_experts

    print(f"\nStatic LFU hit rates:")
    for cap in [100, 500, 1000, 2000, 3000, 5000]:
        if cap > len(sorted_global):
            break
        top_set = set(k for k, _ in sorted_global[:cap])
        hits = sum(1 for t in range(num_tokens) for l in range(num_layers)
                   for ki in range(K) if (l, int(traces[t, l, ki])) in top_set)
        print(f"  C={cap} ({cap/total_blocks*100:.1f}%): {hits/total_accesses*100:.1f}% hit rate")

    # Consecutive overlap
    overlaps = []
    for t in range(1, num_tokens):
        for l in range(num_layers):
            s1 = set(traces[t-1, l].tolist())
            s2 = set(traces[t, l].tolist())
            overlaps.append(len(s1 & s2) / K)
    print(f"\nConsecutive-token overlap: {np.mean(overlaps):.4f} ({np.mean(overlaps)*K:.1f}/{K})")

    # Save stats
    stats = {
        "source": "real_inference_router_logits",
        "model": "Qwen3-30B-A3B",
        "num_tokens": num_tokens,
        "num_layers": num_layers,
        "K": K,
        "mean_zipf_s": round(float(mean_zipf), 4),
        "mean_gini": round(float(mean_gini), 4),
        "mean_consecutive_overlap": round(float(np.mean(overlaps)), 4),
    }
    with open(os.path.join(OUTPUT_DIR, "trace_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n{'='*70}")
    print(f"CRITICAL: Mean Zipf = {mean_zipf:.4f}, Gini = {mean_gini:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    capture_with_single_forward()
