"""
Capture real MoE routing traces from OLMoE-1B-7B.

OLMoE-1B-7B: 6.9B total, 1.3B active, 64 experts, top-8, 16 layers.
At 4-bit: ~1.7 GB — fits entirely in 8 GB VRAM.

This gives us REAL routing traces from actual text inference to validate
our synthetic trace assumptions and measure true Zipf concentration.
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
MODEL_ID = "allenai/OLMoE-1B-7B-0924"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")

PROMPTS = [
    "Explain the theory of general relativity and its implications for modern physics. Einstein's work on spacetime curvature showed that massive objects warp the fabric of space-time itself, affecting how light and matter move through the universe.",
    "Write a Python function that implements the A* pathfinding algorithm with a priority queue. The function should accept a graph represented as an adjacency list and return the shortest path between two nodes.",
    "What are the main causes and effects of climate change on ocean ecosystems? Rising sea temperatures affect coral reefs, marine biodiversity, and ocean circulation patterns throughout the world.",
    "Solve step by step: If f(x) = 3x^2 + 2x - 5, find f'(x) and determine all critical points where f'(x) = 0. Show all intermediate steps.",
    "Tell me a detailed story about a scientist who discovers time travel but faces an ethical dilemma about whether to change the past or preserve the timeline.",
    "Compare and contrast TCP and UDP protocols, including specific scenarios when each should be used in network programming.",
    "Describe the process of photosynthesis at the molecular level, including the light-dependent and light-independent reactions.",
    "Write a SQL query to find the top 5 customers by total order value from the orders and customers tables, including their most recent order date.",
    "What were the key factors that led to the fall of the Roman Empire? Discuss both internal decay and external pressures from barbarian invasions.",
    "Explain how transformer models work in machine learning, focusing on the self-attention mechanism and positional encoding.",
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Flash-MoE V2 -- OLMoE Real Routing Traces")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"Prompts: {len(PROMPTS)}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in 4-bit on GPU
    print("Loading model (4-bit, all GPU)...")
    t0 = time.time()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Check model architecture
    num_layers = model.config.num_hidden_layers
    num_experts = getattr(model.config, 'num_experts', 64)
    K = getattr(model.config, 'num_experts_per_tok', 8)
    print(f"Layers: {num_layers}, Experts: {num_experts}, K: {K}")

    all_traces = []

    for i, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt {i+1}/{len(PROMPTS)} ({len(prompt)} chars) ---")

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        seq_len = inputs["input_ids"].shape[1]
        print(f"  Tokens: {seq_len}")

        t1 = time.time()
        with torch.no_grad():
            try:
                outputs = model(**inputs, output_router_logits=True)
                dt = time.time() - t1
                print(f"  Forward: {dt:.1f}s ({seq_len/dt:.0f} tok/s)")

                if hasattr(outputs, 'router_logits') and outputs.router_logits:
                    router_logits = outputs.router_logits
                    n_layers_out = len(router_logits)
                    print(f"  Router logits: {n_layers_out} layers")

                    traces = np.zeros((seq_len, n_layers_out, K), dtype=np.int32)

                    for l, logits in enumerate(router_logits):
                        if logits is None:
                            continue
                        # logits shape: (batch*seq_len, num_experts)
                        probs = torch.softmax(logits.float(), dim=-1)
                        _, top_idx = torch.topk(probs, K, dim=-1)
                        n = min(seq_len, top_idx.shape[0])
                        traces[:n, l] = top_idx[:n].cpu().numpy()

                    all_traces.append(traces)
                    np.savez_compressed(
                        os.path.join(OUTPUT_DIR, f"olmoe_trace_{i:03d}.npz"),
                        traces=traces, prompt_idx=i
                    )
                    print(f"  Saved: {traces.shape}")
                else:
                    print("  No router_logits! Trying hooks...")
                    # Fallback: use forward hooks
                    layer_data = {}

                    def make_hook(layer_idx):
                        def hook(module, inp, out):
                            if isinstance(out, tuple) and len(out) >= 2:
                                # Try to find expert indices
                                for o in out:
                                    if isinstance(o, torch.Tensor) and o.dtype in (torch.int32, torch.int64, torch.long):
                                        layer_data[layer_idx] = o.detach().cpu().numpy()
                                        break
                        return hook

                    hooks = []
                    for name, module in model.named_modules():
                        if 'gate' in name.lower() or 'router' in name.lower():
                            li = len(hooks)
                            hooks.append(module.register_forward_hook(make_hook(li)))

                    if hooks:
                        print(f"  Attached {len(hooks)} hooks")
                        with torch.no_grad():
                            model(**inputs)
                        print(f"  Hook data layers: {len(layer_data)}")
                        for h in hooks:
                            h.remove()

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

        torch.cuda.empty_cache()

    # Merge and analyze
    if all_traces:
        merged = np.concatenate(all_traces, axis=0)
        np.savez_compressed(os.path.join(OUTPUT_DIR, "olmoe_traces_merged.npz"), traces=merged)
        print(f"\nMerged traces: {merged.shape}")
        analyze(merged, num_experts, K)
    else:
        print("\nNo traces captured!")


def analyze(traces, num_experts, K):
    num_tokens, num_layers, _ = traces.shape
    total_accesses = num_tokens * num_layers * K

    print(f"\n{'='*70}")
    print(f"REAL ROUTING ANALYSIS — OLMoE-1B-7B")
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

    zipf_list, gini_list = [], []
    for l in range(num_layers):
        freqs = sorted(per_layer_freq[l].values(), reverse=True)
        if len(freqs) < 3:
            continue
        total = sum(freqs)
        sf = sorted(freqs)
        n = len(sf)
        gini = (2 * sum((i+1)*f for i, f in enumerate(sf)) - (n+1)*total) / (n*total)
        gini_list.append(gini)

        ranks = np.arange(1, min(31, len(freqs)+1), dtype=np.float64)
        lf = np.log(np.array(freqs[:len(ranks)], dtype=np.float64))
        lr = np.log(ranks)
        if len(lr) >= 3:
            c = np.polyfit(lr, lf, 1)
            zipf_list.append(-c[0])

    mz = np.mean(zipf_list) if zipf_list else 0
    mg = np.mean(gini_list) if gini_list else 0

    print(f"\nZipf exponent: mean={mz:.4f} (min={min(zipf_list):.4f}, max={max(zipf_list):.4f})")
    print(f"Gini: mean={mg:.4f}")

    # Hit rates
    sorted_g = sorted(global_freq.items(), key=lambda x: -x[1])
    total_blocks = num_layers * num_experts
    print(f"\nStatic LFU hit rates (total blocks={total_blocks}):")
    for cap in [50, 100, 200, 500, 800, 1000]:
        if cap > len(sorted_g):
            break
        top = set(k for k, _ in sorted_g[:cap])
        hits = sum(1 for t in range(num_tokens) for l in range(num_layers)
                   for ki in range(K) if (l, int(traces[t, l, ki])) in top)
        print(f"  C={cap} ({cap/total_blocks*100:.1f}%): {hits/total_accesses*100:.1f}%")

    # Overlap
    ovl = []
    for t in range(1, num_tokens):
        for l in range(num_layers):
            s1 = set(traces[t-1, l].tolist())
            s2 = set(traces[t, l].tolist())
            ovl.append(len(s1 & s2) / K)
    mo = np.mean(ovl)
    print(f"\nConsecutive overlap: {mo:.4f} ({mo*K:.1f}/{K})")

    # Max/min ratio
    for l in range(num_layers):
        freqs = sorted(per_layer_freq[l].values(), reverse=True)
        if len(freqs) >= 2:
            ratio = freqs[0] / max(freqs[-1], 1)
            if l in [0, num_layers//2, num_layers-1]:
                print(f"  Layer {l} max/min ratio: {ratio:.1f}x")

    stats = {
        "source": "real_inference",
        "model": "OLMoE-1B-7B",
        "num_tokens": num_tokens,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "K": K,
        "mean_zipf_s": round(float(mz), 4),
        "mean_gini": round(float(mg), 4),
        "mean_overlap": round(float(mo), 4),
        "zipf_per_layer": [round(float(z), 4) for z in zipf_list],
    }
    with open(os.path.join(OUTPUT_DIR, "olmoe_trace_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*70}")
    print(f"CRITICAL: Zipf={mz:.4f}, Gini={mg:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
