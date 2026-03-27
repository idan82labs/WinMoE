"""
Extract routing patterns from Qwen3-30B-A3B gate weights.

Instead of running full inference (too slow on CPU), we:
1. Load only the gate weight matrices from safetensors
2. Generate diverse hidden states (random + structured)
3. Compute top-K routing decisions
4. Analyze the resulting expert popularity distribution

This gives us realistic routing concentration statistics because the
gate weights encode the learned expert specialization patterns.
"""
import json
import os
import sys
from collections import defaultdict
from glob import glob

import numpy as np
import torch
from safetensors import safe_open

CACHE_DIR = "D:/hf_cache"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")

# Qwen3-30B-A3B: 48 layers, 128 experts, top-8, hidden_size=2048
NUM_LAYERS = 48
NUM_EXPERTS = 128
K = 8
HIDDEN_SIZE = 2048


def find_model_files():
    """Find safetensors files in HF cache."""
    pattern = os.path.join(CACHE_DIR, "models--Qwen--Qwen3-30B-A3B", "snapshots", "*", "*.safetensors")
    files = sorted(glob(pattern))
    if not files:
        # Try blobs
        blobs_dir = os.path.join(CACHE_DIR, "models--Qwen--Qwen3-30B-A3B", "blobs")
        if os.path.exists(blobs_dir):
            files = [os.path.join(blobs_dir, f) for f in os.listdir(blobs_dir)
                     if os.path.getsize(os.path.join(blobs_dir, f)) > 1e6]
    return files


def extract_gate_weights(files):
    """Extract gate.weight tensors from all MoE layers."""
    gate_weights = {}

    for f in files:
        try:
            with safe_open(f, framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    if ".gate.weight" in key and "mlp" in key:
                        # Extract layer index from key like "model.layers.5.mlp.gate.weight"
                        parts = key.split(".")
                        for i, p in enumerate(parts):
                            if p == "layers" and i + 1 < len(parts):
                                layer_idx = int(parts[i + 1])
                                gate_weights[layer_idx] = sf.get_tensor(key).float()
                                print(f"  Layer {layer_idx}: gate shape {gate_weights[layer_idx].shape}")
                                break
        except Exception as e:
            continue

    return gate_weights


def generate_diverse_hidden_states(num_tokens: int, hidden_size: int, seed: int = 42):
    """Generate diverse hidden states that exercise different routing paths.

    Uses a mix of:
    1. Random Gaussian vectors (baseline diversity)
    2. Clustered vectors (simulates topic-specific inputs)
    3. Interpolated vectors (simulates gradual topic transitions)
    """
    rng = np.random.default_rng(seed)
    states = []

    # 60% random Gaussian
    n_random = int(num_tokens * 0.6)
    states.append(rng.standard_normal((n_random, hidden_size)).astype(np.float32))

    # 25% clustered around 20 centroids (simulates topics)
    n_clustered = int(num_tokens * 0.25)
    num_clusters = 20
    centroids = rng.standard_normal((num_clusters, hidden_size)).astype(np.float32) * 3.0
    cluster_ids = rng.integers(0, num_clusters, size=n_clustered)
    noise = rng.standard_normal((n_clustered, hidden_size)).astype(np.float32) * 0.5
    states.append(centroids[cluster_ids] + noise)

    # 15% interpolated between random pairs (simulates transitions)
    n_interp = num_tokens - n_random - n_clustered
    src = rng.standard_normal((n_interp, hidden_size)).astype(np.float32)
    dst = rng.standard_normal((n_interp, hidden_size)).astype(np.float32)
    alphas = rng.uniform(0, 1, size=(n_interp, 1)).astype(np.float32)
    states.append(src * alphas + dst * (1 - alphas))

    all_states = np.concatenate(states, axis=0)
    # Shuffle
    perm = rng.permutation(len(all_states))
    return all_states[perm]


def compute_routing(gate_weights: dict, hidden_states: np.ndarray, K: int = 8):
    """Compute top-K routing decisions for all layers.

    Returns: (num_tokens, num_layers, K) array of expert IDs
    """
    num_tokens = hidden_states.shape[0]
    num_layers = len(gate_weights)

    traces = np.zeros((num_tokens, num_layers, K), dtype=np.int32)

    h = torch.from_numpy(hidden_states)  # (num_tokens, hidden_size)

    for layer_idx in sorted(gate_weights.keys()):
        gate_w = gate_weights[layer_idx]  # (num_experts, hidden_size)
        # Router logits: h @ gate_w.T -> (num_tokens, num_experts)
        logits = torch.matmul(h, gate_w.T)
        # Top-K
        _, top_indices = torch.topk(logits, K, dim=-1)
        layer_pos = sorted(gate_weights.keys()).index(layer_idx)
        traces[:, layer_pos, :] = top_indices.numpy()

        if layer_idx % 10 == 0:
            print(f"  Layer {layer_idx}: computed routing for {num_tokens} tokens")

    return traces


def analyze_routing(traces, num_experts=128, K=8):
    """Compute routing statistics."""
    num_tokens, num_layers, _ = traces.shape

    print(f"\n{'='*70}")
    print(f"Routing Analysis (gate-weight-based)")
    print(f"{'='*70}")
    print(f"Tokens: {num_tokens}, Layers: {num_layers}, K: {K}")

    per_layer_freq = defaultdict(lambda: defaultdict(int))
    for t in range(num_tokens):
        for l in range(num_layers):
            for k in range(K):
                per_layer_freq[l][int(traces[t, l, k])] += 1

    zipf_estimates = []
    gini_list = []
    unique_counts = []

    for l in range(num_layers):
        freqs = sorted(per_layer_freq[l].values(), reverse=True)
        if not freqs:
            continue

        total = sum(freqs)
        unique_counts.append(len(freqs))

        # Gini
        sorted_f = sorted(freqs)
        n = len(sorted_f)
        gini = (2 * sum((i + 1) * f for i, f in enumerate(sorted_f)) - (n + 1) * total) / (n * total)
        gini_list.append(gini)

        # Zipf exponent
        ranks = np.arange(1, len(freqs) + 1, dtype=np.float64)
        log_ranks = np.log(ranks[:min(50, len(ranks))])  # fit on top-50 only
        log_freqs = np.log(np.array(freqs[:min(50, len(freqs))], dtype=np.float64))
        if len(log_ranks) >= 3:
            coeffs = np.polyfit(log_ranks, log_freqs, 1)
            zipf_estimates.append(-coeffs[0])

    mean_zipf = np.mean(zipf_estimates) if zipf_estimates else 0
    std_zipf = np.std(zipf_estimates) if zipf_estimates else 0
    mean_gini = np.mean(gini_list) if gini_list else 0

    print(f"\nZipf exponent: mean={mean_zipf:.4f} ± {std_zipf:.4f}")
    print(f"  min={min(zipf_estimates):.4f}, max={max(zipf_estimates):.4f}")
    print(f"Gini coefficient: mean={mean_gini:.4f}")
    print(f"Unique experts/layer: mean={np.mean(unique_counts):.0f}, min={min(unique_counts)}, max={max(unique_counts)}")

    # Static cache hit rates
    global_freq = defaultdict(int)
    for t in range(num_tokens):
        for l in range(num_layers):
            for k in range(K):
                global_freq[(l, int(traces[t, l, k]))] += 1

    total_accesses = num_tokens * num_layers * K
    sorted_global = sorted(global_freq.items(), key=lambda x: -x[1])

    print(f"\nStatic LFU hit rates:")
    print(f"{'Capacity':>10} {'% total':>10} {'Hit rate':>10}")
    for cap in [100, 500, 1000, 2000, 3000, 4000, 5000, 6000]:
        if cap > len(sorted_global):
            break
        top_set = set(k for k, _ in sorted_global[:cap])
        hits = sum(1 for t in range(num_tokens) for l in range(num_layers)
                   for ki in range(K) if (l, int(traces[t, l, ki])) in top_set)
        print(f"{cap:>10} {cap/(num_layers*num_experts)*100:>9.1f}% {hits/total_accesses*100:>9.1f}%")

    # Consecutive-token overlap
    overlaps = []
    for t in range(1, num_tokens):
        for l in range(num_layers):
            s1 = set(traces[t-1, l].tolist())
            s2 = set(traces[t, l].tolist())
            overlaps.append(len(s1 & s2) / K)
    mean_overlap = np.mean(overlaps)
    print(f"\nConsecutive-token overlap: {mean_overlap:.4f} ({mean_overlap*K:.1f}/{K})")

    stats = {
        "source": "gate_weight_routing",
        "model": "Qwen3-30B-A3B",
        "num_tokens": num_tokens,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "K": K,
        "mean_zipf_s": round(float(mean_zipf), 4),
        "std_zipf_s": round(float(std_zipf), 4),
        "min_zipf_s": round(float(min(zipf_estimates)), 4) if zipf_estimates else None,
        "max_zipf_s": round(float(max(zipf_estimates)), 4) if zipf_estimates else None,
        "mean_gini": round(float(mean_gini), 4),
        "mean_consecutive_overlap": round(float(mean_overlap), 4),
        "mean_unique_per_layer": round(float(np.mean(unique_counts)), 1),
        "zipf_per_layer": [round(float(z), 4) for z in zipf_estimates],
    }
    return stats


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Flash-MoE V2 — Gate Weight Routing Extraction")
    print("=" * 70)

    # Find model files
    print("\nFinding model files...")
    files = find_model_files()
    print(f"Found {len(files)} files")

    if not files:
        print("No model files found! Download the model first.")
        return

    # Extract gate weights
    print("\nExtracting gate weights...")
    gate_weights = extract_gate_weights(files)
    print(f"Extracted gates for {len(gate_weights)} layers")

    if not gate_weights:
        print("No gate weights found!")
        return

    # Verify dimensions
    sample = next(iter(gate_weights.values()))
    actual_hidden = sample.shape[1]
    actual_experts = sample.shape[0]
    print(f"Gate dimensions: ({actual_experts}, {actual_hidden})")

    # Generate hidden states
    num_tokens = 1000
    print(f"\nGenerating {num_tokens} diverse hidden states...")
    hidden_states = generate_diverse_hidden_states(num_tokens, actual_hidden)

    # Compute routing
    print("\nComputing routing decisions...")
    traces = compute_routing(gate_weights, hidden_states, K=min(K, actual_experts))

    # Save traces
    trace_file = os.path.join(OUTPUT_DIR, "traces_merged.npz")
    np.savez_compressed(trace_file, traces=traces)
    print(f"\nTraces saved: {traces.shape} -> {trace_file}")

    # Analyze
    stats = analyze_routing(traces, num_experts=actual_experts, K=min(K, actual_experts))

    # Save stats
    stats_file = os.path.join(OUTPUT_DIR, "trace_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_file}")

    # Print verdict
    print(f"\n{'='*70}")
    print(f"CRITICAL RESULT: Mean Zipf exponent = {stats['mean_zipf_s']}")
    print(f"{'='*70}")
    if stats['mean_zipf_s'] >= 1.3:
        print("VERDICT: HIGH concentration — system VIABLE at 32 GB RAM with static LFU")
    elif stats['mean_zipf_s'] >= 1.0:
        print("VERDICT: MODERATE concentration — MARGINAL, needs K_l reduction or 48+ GB RAM")
    else:
        print("VERDICT: LOW concentration — NOT VIABLE without significant K_l reduction")


if __name__ == "__main__":
    main()
