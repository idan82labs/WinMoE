"""
Run cache simulator with real routing traces from Qwen3-30B-A3B.
Automatically picks up traces from work/traces/processed/traces_merged.npz.

Also runs scaling analysis: how do Qwen3-30B results inform Qwen3.5-397B viability?
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from moe_cache_simulator import (
    ModelConfig, TimingParams, compute_expert_frequencies, simulate
)

TRACE_DIR = os.path.join(os.path.dirname(__file__), "..", "traces", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_with_real_traces():
    trace_file = os.path.join(TRACE_DIR, "traces_merged.npz")
    stats_file = os.path.join(TRACE_DIR, "trace_stats.json")

    if not os.path.exists(trace_file):
        print(f"No traces found at {trace_file}")
        print("Run capture_real_traces.py first.")
        return None

    # Load traces
    data = np.load(trace_file)
    traces = data["traces"]
    num_tokens, num_layers, K = traces.shape
    print(f"Loaded real traces: {traces.shape} (tokens={num_tokens}, layers={num_layers}, K={K})")

    # Load stats if available
    if os.path.exists(stats_file):
        with open(stats_file) as f:
            stats = json.load(f)
        print(f"Zipf exponent: {stats.get('mean_zipf_s', 'N/A')}")
        print(f"Gini: {stats.get('mean_gini', 'N/A')}")
        print(f"Token overlap: {stats.get('mean_consecutive_overlap', 'N/A')}")

    # Qwen3-30B-A3B config
    model_30b = ModelConfig(
        num_layers=num_layers,  # should be 48
        num_experts=128,
        num_active=K,            # should be 8
        has_shared=False,
        expert_size_bytes=2_359_296,  # ~2.36 MB at Q4 (4.72M params * 0.5)
    )

    timing = TimingParams()

    # Compute frequencies
    print("\nComputing expert frequencies...")
    freqs = compute_expert_frequencies(traces, num_layers, 128)
    total_blocks = num_layers * 128
    print(f"Total expert blocks: {total_blocks}")
    print(f"Unique accessed: {len(freqs)}")

    # ── Sweep 1: RAM capacity ──
    vram_cap = 500  # ~1.18 GB — small VRAM allocation for 30B
    ram_caps = [0, 500, 1000, 2000, 3000, 4000, 5000, 6000]

    print(f"\n{'='*80}")
    print(f"RAM Capacity Sweep — Qwen3-30B-A3B Real Traces (VRAM={vram_cap})")
    print(f"{'='*80}")
    print(f"{'RAM':>8} {'RAM GB':>8} {'VRAM%':>8} {'RAM%':>8} {'SSD%':>8} {'MeanSvc':>10} {'P95':>10} {'rho':>8}")
    print("-" * 80)

    ram_results = []
    for rc in ram_caps:
        r = simulate(traces, model_30b, timing, vram_cap, rc, "static_lfu", freqs)
        s = r.summary()
        ram_gb = rc * model_30b.expert_size_bytes / 1e9
        rho = s["mean_service_ms"] / (timing.T_c_s * 1000)
        print(f"{rc:>8} {ram_gb:>7.1f} {s['vram_hit%']:>7.1f}% {s['ram_hit%']:>7.1f}% "
              f"{s['ssd_miss%']:>7.1f}% {s['mean_service_ms']:>9.2f}ms "
              f"{s['p95_service_ms']:>9.2f}ms {rho:>7.3f}")
        s["ram_cap"] = rc
        s["ram_gb"] = round(ram_gb, 1)
        s["rho"] = round(rho, 3)
        ram_results.append(s)

    # ── Sweep 2: K reduction ──
    print(f"\n{'='*80}")
    print(f"K Reduction (simulated by truncating top-K) — VRAM={vram_cap}, RAM=3000")
    print(f"{'='*80}")

    ram_cap_fixed = 3000
    k_results = []
    for K_new in [2, 3, 4, 5, 6, 7, 8]:
        if K_new > K:
            continue
        # Truncate traces to first K_new experts
        truncated = traces[:, :, :K_new]
        model_k = ModelConfig(
            num_layers=num_layers, num_experts=128, num_active=K_new,
            has_shared=False, expert_size_bytes=model_30b.expert_size_bytes,
        )
        freqs_k = compute_expert_frequencies(truncated, num_layers, 128)
        r = simulate(truncated, model_k, timing, vram_cap, ram_cap_fixed, "static_lfu", freqs_k)
        s = r.summary()
        rho = s["mean_service_ms"] / (timing.T_c_s * 1000)
        print(f"  K={K_new}: SSD={s['ssd_miss%']:.1f}% mean_svc={s['mean_service_ms']:.2f}ms rho={rho:.3f}")
        s["K"] = K_new
        s["rho"] = round(rho, 3)
        k_results.append(s)

    # ── Scaling to 397B ──
    print(f"\n{'='*80}")
    print(f"Scaling projection: Qwen3-30B -> Qwen3.5-397B")
    print(f"{'='*80}")

    # Key difference: 397B has 512 experts, K=10+1, 60 layers, 6.29 MB/expert
    # The Zipf concentration measured on 30B applies to 397B with adjustment
    # for having 4x more experts per layer (512 vs 128)

    if os.path.exists(stats_file):
        with open(stats_file) as f:
            stats = json.load(f)
        zipf_s = stats.get("mean_zipf_s", 1.2)
        print(f"Measured Zipf exponent from 30B: {zipf_s}")
        print(f"\nWith 512 experts (vs 128), the cache percentage of total is 4x smaller.")
        print(f"However, if Zipf exponent is preserved, the top-N concentration is similar.")
        print(f"\nProjected viability for 397B at zipf_s={zipf_s}:")
        print(f"  - If zipf_s >= 1.3: VIABLE with 32 GB RAM + K_l=7-8")
        print(f"  - If zipf_s in [1.0, 1.3]: MARGINAL, needs K_l <= 7 or 48+ GB RAM")
        print(f"  - If zipf_s < 1.0: NOT VIABLE without major K_l reduction")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "model": "Qwen3-30B-A3B",
        "trace_shape": list(traces.shape),
        "ram_sweep": ram_results,
        "k_reduction": k_results,
    }
    out_file = os.path.join(RESULTS_DIR, "real_trace_results.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_file}")

    return output


if __name__ == "__main__":
    run_with_real_traces()
