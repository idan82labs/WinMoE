"""
Capture real routing traces from Qwen3-30B-A3B using bitsandbytes 4-bit.

Downloads model to D:/hf_cache, loads in 4-bit (~18 GB RAM),
captures per-token per-layer expert routing via forward hooks.
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CACHE_DIR = "D:/hf_cache"
MODEL_ID = "Qwen/Qwen3-30B-A3B"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")

PROMPTS = [
    "Explain the theory of general relativity and its implications for modern physics.",
    "Write a Python function that implements the A* pathfinding algorithm with a priority queue.",
    "What are the main causes and effects of climate change on ocean ecosystems?",
]


class RouterTracer:
    """Captures MoE routing decisions via forward hooks."""

    def __init__(self):
        self.layer_traces = {}  # layer_idx -> list of (expert_ids, gate_weights) per token
        self._hooks = []
        self.num_layers = 0

    def attach(self, model):
        layer_idx = 0
        for name, module in model.named_modules():
            if "gate" in name and hasattr(module, "weight") and name.endswith(".gate"):
                self._hooks.append(
                    module.register_forward_hook(self._make_hook(layer_idx))
                )
                self.layer_traces[layer_idx] = []
                layer_idx += 1
        self.num_layers = layer_idx
        print(f"Attached hooks to {layer_idx} MoE layers")

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                if len(output) >= 3:
                    logits, weights, indices = output[0], output[1], output[2]
                elif len(output) == 2:
                    logits, indices = output
                    weights = None
                else:
                    return

                # indices: [batch*seq, top_k] — expert IDs selected
                # weights: [batch*seq, top_k] — gate weights
                ids = indices.detach().cpu().numpy()
                w = weights.detach().cpu().numpy() if weights is not None else None
                self.layer_traces[layer_idx].append((ids, w))
        return hook_fn

    def clear(self):
        for k in self.layer_traces:
            self.layer_traces[k] = []

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def get_traces_array(self):
        """Build (total_tokens, num_layers, K) array from captured data."""
        if not self.layer_traces or not self.layer_traces[0]:
            return None, None

        # Each forward pass captures (seq_len, K) for each layer
        # Concatenate across forward passes
        all_ids = {}
        all_weights = {}
        for layer_idx in range(self.num_layers):
            entries = self.layer_traces[layer_idx]
            ids_list = [e[0] for e in entries]  # list of (seq, K)
            w_list = [e[1] for e in entries if e[1] is not None]

            all_ids[layer_idx] = np.concatenate(ids_list, axis=0)  # (total_tokens, K)
            if w_list:
                all_weights[layer_idx] = np.concatenate(w_list, axis=0)

        # Stack layers: (num_layers, total_tokens, K) -> (total_tokens, num_layers, K)
        num_tokens = all_ids[0].shape[0]
        K = all_ids[0].shape[1]

        traces = np.zeros((num_tokens, self.num_layers, K), dtype=np.int32)
        weights = np.zeros((num_tokens, self.num_layers, K), dtype=np.float32) if all_weights else None

        for l in range(self.num_layers):
            n = min(num_tokens, all_ids[l].shape[0])
            traces[:n, l, :] = all_ids[l][:n]
            if weights is not None and l in all_weights:
                n_w = min(num_tokens, all_weights[l].shape[0])
                weights[:n_w, l, :] = all_weights[l][:n_w]

        return traces, weights


def analyze_traces(traces, num_experts=128, K=8):
    """Compute routing statistics from captured traces."""
    num_tokens, num_layers, _ = traces.shape
    total_accesses = num_tokens * num_layers * K

    print(f"\n{'='*70}")
    print(f"Routing Trace Analysis")
    print(f"{'='*70}")
    print(f"Tokens: {num_tokens}, Layers: {num_layers}, K: {K}")
    print(f"Total expert accesses: {total_accesses}")

    # Global expert frequency
    global_freq = defaultdict(int)
    per_layer_freq = defaultdict(lambda: defaultdict(int))

    for t in range(num_tokens):
        for l in range(num_layers):
            for k in range(K):
                eid = int(traces[t, l, k])
                global_freq[(l, eid)] += 1
                per_layer_freq[l][eid] += 1

    # Compute Zipf-like concentration per layer
    zipf_estimates = []
    layer_ginis = []

    for l in range(num_layers):
        freqs = sorted(per_layer_freq[l].values(), reverse=True)
        if not freqs:
            continue

        total = sum(freqs)
        probs = [f / total for f in freqs]

        # Gini coefficient
        n = len(probs)
        sorted_p = sorted(probs)
        gini = (2 * sum((i + 1) * p for i, p in enumerate(sorted_p)) - (n + 1) * sum(sorted_p)) / (n * sum(sorted_p)) if sum(sorted_p) > 0 else 0
        layer_ginis.append(gini)

        # Estimate Zipf exponent via log-log regression
        ranks = np.arange(1, len(freqs) + 1, dtype=np.float64)
        log_ranks = np.log(ranks)
        log_freqs = np.log(np.array(freqs, dtype=np.float64) + 1e-10)

        # Linear fit: log(freq) = -s * log(rank) + c
        if len(log_ranks) >= 3:
            coeffs = np.polyfit(log_ranks, log_freqs, 1)
            zipf_s = -coeffs[0]
            zipf_estimates.append(zipf_s)

    mean_zipf = np.mean(zipf_estimates) if zipf_estimates else 0
    mean_gini = np.mean(layer_ginis) if layer_ginis else 0

    print(f"\nPer-layer Zipf exponent: mean={mean_zipf:.3f}, min={min(zipf_estimates):.3f}, max={max(zipf_estimates):.3f}")
    print(f"Per-layer Gini coefficient: mean={mean_gini:.3f}")

    # Top-C hit rates (simulate static cache)
    total_blocks = num_layers * num_experts
    sorted_global = sorted(global_freq.items(), key=lambda x: -x[1])

    capacities = [100, 500, 1000, 2000, 3000, 5000, 7500, 10000]
    print(f"\nStatic LFU hit rates:")
    print(f"{'Capacity':>10} {'% of total':>12} {'Hit rate':>10}")
    print("-" * 35)

    for cap in capacities:
        if cap > len(sorted_global):
            break
        top_set = set(k for k, _ in sorted_global[:cap])
        hits = sum(1 for t in range(num_tokens) for l in range(num_layers)
                   for k_idx in range(K) if (l, int(traces[t, l, k_idx])) in top_set)
        hit_rate = hits / total_accesses
        print(f"{cap:>10} {cap/total_blocks*100:>11.1f}% {hit_rate*100:>9.1f}%")

    # Consecutive-token overlap
    overlaps = []
    for t in range(1, num_tokens):
        for l in range(num_layers):
            set_prev = set(traces[t-1, l].tolist())
            set_curr = set(traces[t, l].tolist())
            overlap = len(set_prev & set_curr) / K
            overlaps.append(overlap)

    mean_overlap = np.mean(overlaps) if overlaps else 0
    print(f"\nConsecutive-token overlap: {mean_overlap:.3f} ({mean_overlap*K:.1f}/{K} experts shared)")

    # Unique experts per layer
    unique_per_layer = []
    for l in range(num_layers):
        unique = len(set(traces[:, l, :].flatten().tolist()))
        unique_per_layer.append(unique)
    print(f"Unique experts per layer: mean={np.mean(unique_per_layer):.0f}, min={min(unique_per_layer)}, max={max(unique_per_layer)} (out of {num_experts})")

    stats = {
        "num_tokens": num_tokens,
        "num_layers": num_layers,
        "K": K,
        "mean_zipf_s": round(float(mean_zipf), 4),
        "min_zipf_s": round(float(min(zipf_estimates)), 4) if zipf_estimates else None,
        "max_zipf_s": round(float(max(zipf_estimates)), 4) if zipf_estimates else None,
        "mean_gini": round(float(mean_gini), 4),
        "mean_consecutive_overlap": round(float(mean_overlap), 4),
        "mean_unique_per_layer": round(float(np.mean(unique_per_layer)), 1),
        "zipf_per_layer": [round(float(z), 4) for z in zipf_estimates],
    }
    return stats


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("=" * 70)
    print("Flash-MoE V2 — Real Routing Trace Capture")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"Cache: {CACHE_DIR}")
    print(f"Prompts: {len(PROMPTS)}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in 4-bit with bitsandbytes
    print("Loading model in 4-bit quantization...")
    t0 = time.time()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": "cpu"},
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Attach tracer
    tracer = RouterTracer()
    tracer.attach(model)

    # Generate with tracing — reduced for CPU speed
    max_new_tokens = 10

    for i, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt {i+1}/{len(PROMPTS)} ---")
        print(f"  {prompt[:80]}...")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            # Prefill: process all input tokens at once
            tracer.clear()

            for step in range(max_new_tokens):
                outputs = model(input_ids)
                next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

            traces, weights = tracer.get_traces_array()
            if traces is not None:
                print(f"  Captured: {traces.shape[0]} tokens × {traces.shape[1]} layers × {traces.shape[2]} experts")

                # Save per-prompt trace
                prompt_file = os.path.join(OUTPUT_DIR, f"trace_prompt_{i:03d}.npz")
                save_dict = {"traces": traces, "prompt_idx": i}
                if weights is not None:
                    save_dict["gate_weights"] = weights
                np.savez_compressed(prompt_file, **save_dict)

    # Merge all traces
    print("\n\nMerging all traces...")
    all_traces = []
    all_weights = []
    for i in range(len(PROMPTS)):
        f = os.path.join(OUTPUT_DIR, f"trace_prompt_{i:03d}.npz")
        if os.path.exists(f):
            data = np.load(f)
            all_traces.append(data["traces"])
            if "gate_weights" in data:
                all_weights.append(data["gate_weights"])

    if all_traces:
        merged_traces = np.concatenate(all_traces, axis=0)
        merged_file = os.path.join(OUTPUT_DIR, "traces_merged.npz")
        save_dict = {"traces": merged_traces}
        if all_weights:
            merged_weights = np.concatenate(all_weights, axis=0)
            save_dict["gate_weights"] = merged_weights
        np.savez_compressed(merged_file, **save_dict)
        print(f"Merged traces: {merged_traces.shape}")

        # Analyze
        stats = analyze_traces(merged_traces, num_experts=128, K=8)

        # Save stats
        stats_file = os.path.join(OUTPUT_DIR, "trace_stats.json")
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats saved to {stats_file}")

        # Print the critical number
        print(f"\n{'='*70}")
        print(f"CRITICAL RESULT: Mean Zipf exponent = {stats['mean_zipf_s']}")
        print(f"{'='*70}")
        if stats['mean_zipf_s'] >= 1.3:
            print("VERDICT: Zipf concentration is HIGH ENOUGH for viability at 32 GB RAM")
        elif stats['mean_zipf_s'] >= 1.0:
            print("VERDICT: MARGINAL — may need K_l reduction or more RAM")
        else:
            print("VERDICT: LOW concentration — viability requires significant K_l reduction or 64+ GB RAM")
    else:
        print("No traces captured!")

    tracer.detach()
    print("\nDone.")


if __name__ == "__main__":
    main()
