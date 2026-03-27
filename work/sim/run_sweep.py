"""Run comprehensive cache parameter sweeps for Phase 0 analysis."""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from moe_cache_simulator import (
    ModelConfig, TimingParams, generate_synthetic_trace,
    compute_expert_frequencies, simulate
)


def run_ram_sweep():
    """Sweep RAM capacity at fixed VRAM to find critical RAM size."""
    model = ModelConfig()
    timing = TimingParams()

    print("Generating synthetic trace (500 tokens, zipf_s=1.2)...")
    traces = generate_synthetic_trace(500, model, zipf_s=1.2, seed=42)
    freqs = compute_expert_frequencies(traces, model.num_layers, model.num_experts)

    total_blocks = model.num_layers * model.num_experts
    print(f"Total expert blocks: {total_blocks}")
    print(f"Unique accessed: {len(freqs)}")

    # VRAM fixed at ~1000 blocks (realistic for 8GB GPU after overhead)
    vram_cap = 1000

    # RAM sweep: from 0 to 15000 blocks (0 to ~94 GB)
    ram_caps = [0, 1000, 2000, 3000, 5000, 7500, 10000, 12000, 15000]

    print(f"\n{'='*80}")
    print(f"RAM Capacity Sweep (VRAM={vram_cap} blocks fixed, static_lfu)")
    print(f"{'='*80}")
    print(f"{'RAM blocks':>12} {'RAM GB':>8} {'VRAM%':>8} {'RAM%':>8} {'SSD%':>8} "
          f"{'MeanSvc':>10} {'P95':>10} {'rho':>8} {'Status':>12}")
    print("-" * 100)

    results = []
    for rc in ram_caps:
        r = simulate(traces, model, timing, vram_cap, rc, "static_lfu", freqs)
        s = r.summary()
        ram_gb = rc * model.expert_size_bytes / 1e9
        rho = s["mean_service_ms"] / (timing.T_c_s * 1000)
        status = "VIABLE" if rho < 1.0 else ("MARGINAL" if rho < 1.5 else "NOT VIABLE")
        print(f"{rc:>12} {ram_gb:>7.1f} {s['vram_hit%']:>7.1f}% {s['ram_hit%']:>7.1f}% "
              f"{s['ssd_miss%']:>7.1f}% {s['mean_service_ms']:>9.2f}ms "
              f"{s['p95_service_ms']:>9.2f}ms {rho:>7.3f} {status:>12}")
        s["ram_gb"] = round(ram_gb, 1)
        s["rho"] = round(rho, 3)
        s["status"] = status
        results.append(s)

    return results


def run_zipf_sensitivity():
    """Test sensitivity to Zipf concentration."""
    model = ModelConfig()
    timing = TimingParams()
    vram_cap = 1000
    ram_cap = 5000  # ~31.5 GB

    print(f"\n{'='*80}")
    print(f"Zipf Exponent Sensitivity (VRAM={vram_cap}, RAM={ram_cap})")
    print(f"{'='*80}")
    print(f"{'zipf_s':>8} {'VRAM%':>8} {'RAM%':>8} {'SSD%':>8} {'MeanSvc':>10} "
          f"{'P95':>10} {'rho':>8} {'Status':>12}")
    print("-" * 90)

    results = []
    for zipf_s in [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]:
        traces = generate_synthetic_trace(500, model, zipf_s=zipf_s, seed=42)
        freqs = compute_expert_frequencies(traces, model.num_layers, model.num_experts)
        r = simulate(traces, model, timing, vram_cap, ram_cap, "static_lfu", freqs)
        s = r.summary()
        rho = s["mean_service_ms"] / (timing.T_c_s * 1000)
        status = "VIABLE" if rho < 1.0 else ("MARGINAL" if rho < 1.5 else "NOT VIABLE")
        print(f"{zipf_s:>8.1f} {s['vram_hit%']:>7.1f}% {s['ram_hit%']:>7.1f}% "
              f"{s['ssd_miss%']:>7.1f}% {s['mean_service_ms']:>9.2f}ms "
              f"{s['p95_service_ms']:>9.2f}ms {rho:>7.3f} {status:>12}")
        s["zipf_s"] = zipf_s
        s["rho"] = round(rho, 3)
        s["status"] = status
        results.append(s)

    return results


def run_token_length_sensitivity():
    """Test how trace length affects cacheability."""
    model = ModelConfig()
    timing = TimingParams()
    vram_cap = 1000
    ram_cap = 5000

    print(f"\n{'='*80}")
    print(f"Token Length Sensitivity (VRAM={vram_cap}, RAM={ram_cap}, zipf=1.2)")
    print(f"{'='*80}")
    print(f"{'Tokens':>8} {'Unique':>8} {'VRAM%':>8} {'RAM%':>8} {'SSD%':>8} "
          f"{'MeanSvc':>10} {'rho':>8}")
    print("-" * 70)

    for num_tokens in [50, 100, 200, 500, 1000]:
        traces = generate_synthetic_trace(num_tokens, model, zipf_s=1.2, seed=42)
        freqs = compute_expert_frequencies(traces, model.num_layers, model.num_experts)
        r = simulate(traces, model, timing, vram_cap, ram_cap, "static_lfu", freqs)
        s = r.summary()
        rho = s["mean_service_ms"] / (timing.T_c_s * 1000)
        print(f"{num_tokens:>8} {len(freqs):>8} {s['vram_hit%']:>7.1f}% {s['ram_hit%']:>7.1f}% "
              f"{s['ssd_miss%']:>7.1f}% {s['mean_service_ms']:>9.2f}ms {rho:>7.3f}")


if __name__ == "__main__":
    print("Flash-MoE V2 — Comprehensive Cache Parameter Sweep")
    print("=" * 80)

    ram_results = run_ram_sweep()
    zipf_results = run_zipf_sensitivity()
    run_token_length_sensitivity()

    # Save all results
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ram_sweep.json"), "w") as f:
        json.dump(ram_results, f, indent=2)
    with open(os.path.join(out_dir, "zipf_sweep.json"), "w") as f:
        json.dump(zipf_results, f, indent=2)
    print(f"\nSweep results saved to {out_dir}")
