"""
Flash-MoE V2 — K Reduction and Layerwise K_l Analysis

Models the effect of reducing active expert count per layer:
  - Lower K → fewer SSD reads per layer → lower miss-service demand
  - But lower K → potential quality degradation (retained mass proxy)

Produces:
  - Cost-vs-quality surrogate curves
  - Viability region shifts under different K values
  - Recommended K_l schedules
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from moe_cache_simulator import (
    ModelConfig, TimingParams, generate_synthetic_trace,
    compute_expert_frequencies, simulate
)


def compute_retained_mass(traces: np.ndarray, K_orig: int, K_new: int) -> float:
    """Compute retained mass p(K_new) = fraction of original gate weight retained.

    Since we use top-K routing, the retained mass when reducing from K_orig to K_new
    is the fraction of accesses that would still be selected.

    For synthetic traces without gate weights, we approximate by assuming
    Zipf-distributed gate weights within each layer's selected experts.
    """
    if K_new >= K_orig:
        return 1.0
    # Approximate: the top K_new out of K_orig retain (K_new/K_orig) of accesses
    # Under Zipf within the selected set, it's actually higher
    # Simple proxy: sum of top K_new Zipf weights / sum of K_orig Zipf weights
    ranks = np.arange(1, K_orig + 1, dtype=np.float64)
    weights = 1.0 / ranks  # harmonic weights
    total = weights.sum()
    retained = weights[:K_new].sum()
    return retained / total


def simulate_at_k(K_new: int, model_config: ModelConfig, timing: TimingParams,
                   vram_cap: int, ram_cap: int, num_tokens: int = 500,
                   zipf_s: float = 1.2, seed: int = 42):
    """Run simulation with reduced K value."""
    # Generate traces with K_new active experts
    model_k = ModelConfig(
        num_layers=model_config.num_layers,
        num_experts=model_config.num_experts,
        num_active=K_new,
        has_shared=model_config.has_shared,
        expert_size_bytes=model_config.expert_size_bytes,
        shared_expert_size_bytes=model_config.shared_expert_size_bytes,
    )
    traces = generate_synthetic_trace(num_tokens, model_k, zipf_s=zipf_s, seed=seed)
    freqs = compute_expert_frequencies(traces, model_k.num_layers, model_k.num_experts)
    result = simulate(traces, model_k, timing, vram_cap, ram_cap, "static_lfu", freqs)
    return result


def run_k_sweep():
    """Sweep K from 2 to 10 and measure viability impact."""
    model = ModelConfig()
    timing = TimingParams()
    vram_cap = 1000
    ram_cap = 5000  # ~31.5 GB

    K_orig = model.num_active
    K_values = [2, 3, 4, 5, 6, 7, 8, 10]

    print(f"{'='*90}")
    print(f"K Reduction Sweep (VRAM={vram_cap}, RAM={ram_cap}, zipf=1.2)")
    print(f"{'='*90}")
    print(f"{'K':>4} {'Retained':>10} {'MissMB/lyr':>12} {'SSD%':>8} {'MeanSvc':>10} "
          f"{'P95':>10} {'rho':>8} {'Status':>12}")
    print("-" * 90)

    results = []
    for K in K_values:
        r = simulate_at_k(K, model, timing, vram_cap, ram_cap)
        s = r.summary()
        rho = s["mean_service_ms"] / (timing.T_c_s * 1000)
        retained = compute_retained_mass(None, K_orig, K)
        miss_mb_per_layer = K * model.expert_size_bytes / 1e6
        status = "VIABLE" if rho < 1.0 else ("MARGINAL" if rho < 1.5 else "NOT VIABLE")

        print(f"{K:>4} {retained:>9.1%} {miss_mb_per_layer:>11.1f} {s['ssd_miss%']:>7.1f}% "
              f"{s['mean_service_ms']:>9.2f}ms {s['p95_service_ms']:>9.2f}ms "
              f"{rho:>7.3f} {status:>12}")

        results.append({
            "K": K,
            "retained_mass": round(retained, 4),
            "miss_mb_per_layer": round(miss_mb_per_layer, 1),
            "ssd_miss_pct": s["ssd_miss%"],
            "mean_service_ms": s["mean_service_ms"],
            "p95_service_ms": s["p95_service_ms"],
            "rho": round(rho, 3),
            "status": status,
        })

    return results


def run_k_vs_ram_matrix():
    """Matrix of K × RAM to find optimal operating points."""
    model = ModelConfig()
    timing = TimingParams()
    vram_cap = 1000

    K_values = [4, 6, 8, 10]
    ram_caps = [3000, 5000, 7500, 10000]

    print(f"\n{'='*90}")
    print(f"K × RAM Viability Matrix (rho values, VRAM={vram_cap})")
    print(f"{'='*90}")

    # Header
    header = f"{'K':>4} |"
    for rc in ram_caps:
        ram_gb = rc * model.expert_size_bytes / 1e9
        header += f" RAM={ram_gb:.0f}GB |"
    print(header)
    print("-" * len(header))

    matrix = {}
    for K in K_values:
        row = f"{K:>4} |"
        for rc in ram_caps:
            r = simulate_at_k(K, model, timing, vram_cap, rc)
            s = r.summary()
            rho = s["mean_service_ms"] / (timing.T_c_s * 1000)
            status_char = "OK" if rho < 1.0 else ("~" if rho < 1.5 else "X")
            row += f" {rho:>5.2f} [{status_char:>2}] |"
            matrix[(K, rc)] = rho
        print(row)

    return matrix


if __name__ == "__main__":
    print("Flash-MoE V2 — K Reduction Analysis")
    print("=" * 90)

    k_results = run_k_sweep()
    matrix = run_k_vs_ram_matrix()

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "k_reduction.json"), "w") as f:
        json.dump(k_results, f, indent=2)
    print(f"\nResults saved to {out_dir}")
