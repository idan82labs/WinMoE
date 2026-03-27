"""
Flash-MoE Windows V2 — RAM→GPU (PCIe) Transfer Benchmark

Measures:
  1. Pinned vs pageable RAM→GPU transfer at varying sizes
  2. Per-transfer overhead at varying request counts
  3. cudaMemcpyAsync staging patterns
  4. GPU→RAM (reverse direction) for comparison

Produces timing data for affine model:
  t(x, m) = alpha + beta*m + x/B
"""

import argparse
import json
import os
import sys
import time

import numpy as np

try:
    import torch
    assert torch.cuda.is_available()
except (ImportError, AssertionError):
    print("PyTorch with CUDA required.")
    sys.exit(1)


def bench_h2d_pageable(size_bytes: int, num_transfers: int = 1, warmup: int = 3) -> dict:
    """Host→Device transfer from pageable (regular) memory."""
    src = torch.randn(size_bytes // 4, dtype=torch.float32)  # pageable
    dst = torch.empty(size_bytes // 4, dtype=torch.float32, device="cuda")

    # Warmup
    for _ in range(warmup):
        dst.copy_(src)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(num_transfers):
        dst.copy_(src)
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()

    elapsed_ns = t1 - t0
    total_bytes = size_bytes * num_transfers
    elapsed_s = elapsed_ns / 1e9
    bw = (total_bytes / (1024**2)) / elapsed_s if elapsed_s > 0 else 0

    del src, dst
    torch.cuda.empty_cache()

    return {
        "method": "h2d_pageable",
        "transfer_size": size_bytes,
        "num_transfers": num_transfers,
        "total_bytes": total_bytes,
        "elapsed_ns": elapsed_ns,
        "elapsed_ms": round(elapsed_ns / 1e6, 3),
        "bandwidth_MBps": round(bw, 1),
        "per_transfer_us": round((elapsed_ns / num_transfers) / 1000, 2),
    }


def bench_h2d_pinned(size_bytes: int, num_transfers: int = 1, warmup: int = 3) -> dict:
    """Host→Device transfer from pinned (page-locked) memory."""
    src = torch.randn(size_bytes // 4, dtype=torch.float32).pin_memory()
    dst = torch.empty(size_bytes // 4, dtype=torch.float32, device="cuda")

    for _ in range(warmup):
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(num_transfers):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()

    elapsed_ns = t1 - t0
    total_bytes = size_bytes * num_transfers
    elapsed_s = elapsed_ns / 1e9
    bw = (total_bytes / (1024**2)) / elapsed_s if elapsed_s > 0 else 0

    del src, dst
    torch.cuda.empty_cache()

    return {
        "method": "h2d_pinned",
        "transfer_size": size_bytes,
        "num_transfers": num_transfers,
        "total_bytes": total_bytes,
        "elapsed_ns": elapsed_ns,
        "elapsed_ms": round(elapsed_ns / 1e6, 3),
        "bandwidth_MBps": round(bw, 1),
        "per_transfer_us": round((elapsed_ns / num_transfers) / 1000, 2),
    }


def bench_d2h_pinned(size_bytes: int, num_transfers: int = 1, warmup: int = 3) -> dict:
    """Device→Host transfer to pinned memory (for comparison)."""
    src = torch.randn(size_bytes // 4, dtype=torch.float32, device="cuda")
    dst = torch.empty(size_bytes // 4, dtype=torch.float32).pin_memory()

    for _ in range(warmup):
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(num_transfers):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()

    elapsed_ns = t1 - t0
    total_bytes = size_bytes * num_transfers
    elapsed_s = elapsed_ns / 1e9
    bw = (total_bytes / (1024**2)) / elapsed_s if elapsed_s > 0 else 0

    del src, dst
    torch.cuda.empty_cache()

    return {
        "method": "d2h_pinned",
        "transfer_size": size_bytes,
        "num_transfers": num_transfers,
        "total_bytes": total_bytes,
        "elapsed_ns": elapsed_ns,
        "elapsed_ms": round(elapsed_ns / 1e6, 3),
        "bandwidth_MBps": round(bw, 1),
        "per_transfer_us": round((elapsed_ns / num_transfers) / 1000, 2),
    }


def run_size_sweep():
    """Sweep transfer sizes to measure bandwidth vs size."""
    sizes = [
        4 * 1024,           # 4 KB
        16 * 1024,          # 16 KB
        64 * 1024,          # 64 KB
        256 * 1024,         # 256 KB
        1 * 1024 * 1024,    # 1 MB
        4 * 1024 * 1024,    # 4 MB
        16 * 1024 * 1024,   # 16 MB
        64 * 1024 * 1024,   # 64 MB
        128 * 1024 * 1024,  # 128 MB
    ]
    results = []
    for sz in sizes:
        sz_str = f"{sz // 1024} KB" if sz < 1024 * 1024 else f"{sz // (1024*1024)} MB"
        print(f"  Size {sz_str}...")
        # Multiple repeats for small sizes
        reps = max(1, min(100, 64 * 1024 * 1024 // sz))
        results.append(bench_h2d_pinned(sz, reps))
        results.append(bench_h2d_pageable(sz, reps))
    return results


def run_count_sweep(fixed_total_mb: int = 64):
    """Fixed total bytes, varying transfer count to isolate per-transfer overhead."""
    total = fixed_total_mb * 1024 * 1024
    configs = [
        (total, 1),
        (total // 4, 4),
        (total // 16, 16),
        (total // 64, 64),
        (total // 256, 256),
        (total // 1024, 1024),
    ]
    results = []
    for xfer_size, count in configs:
        if xfer_size < 4096:
            continue
        # Align to 4 bytes (float32)
        xfer_size = (xfer_size // 4) * 4
        print(f"  {count} x {xfer_size // 1024} KB...")
        results.append(bench_h2d_pinned(xfer_size, count))
    return results


def fit_affine(results: list, method_filter: str = "h2d_pinned") -> dict:
    """Fit affine model from count sweep data."""
    pts = [r for r in results if r["method"] == method_filter and "sweep" not in r]
    # Use count sweep results (those with varying num_transfers)
    count_pts = [r for r in pts if r["num_transfers"] > 0]
    if len(count_pts) < 3:
        return {"error": "not enough points"}

    A = []
    b = []
    for p in count_pts:
        m = p["num_transfers"]
        x = p["total_bytes"]
        t = p["elapsed_ns"] / 1e9
        A.append([1.0, float(m), float(x)])
        b.append(t)

    A_arr = np.array(A)
    b_arr = np.array(b)
    result, _, _, _ = np.linalg.lstsq(A_arr, b_arr, rcond=None)
    alpha, beta, inv_B = result
    B = 1.0 / inv_B if abs(inv_B) > 1e-15 else float("inf")
    x_star = alpha * B if B != float("inf") else float("nan")

    return {
        "alpha_us": round(alpha * 1e6, 2),
        "beta_us": round(beta * 1e6, 2),
        "B_MBps": round(B / (1024 * 1024), 1),
        "B_GBps": round(B / (1024**3), 3),
        "x_star_KB": round(x_star / 1024, 1) if not np.isnan(x_star) else "N/A",
        "num_points": len(count_pts),
    }


def print_summary(results: list):
    print("\n" + "=" * 90)
    print(f"{'Method':<20} {'XferSize':>10} {'Count':>6} {'TotalMB':>8} {'ms':>10} {'MB/s':>10} {'us/xfer':>10}")
    print("-" * 90)
    for r in results:
        sz = r["transfer_size"]
        sz_str = f"{sz // 1024}K" if sz < 1024 * 1024 else f"{sz // (1024*1024)}M"
        total_mb = r["total_bytes"] / (1024 * 1024)
        print(f"{r['method']:<20} {sz_str:>10} {r['num_transfers']:>6} {total_mb:>8.1f} "
              f"{r['elapsed_ms']:>10.2f} {r['bandwidth_MBps']:>10.1f} {r['per_transfer_us']:>10.1f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="PCIe RAM→GPU Benchmark")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), ".."))
    args = parser.parse_args()

    print("=" * 60)
    print("Flash-MoE Windows V2 — PCIe Transfer Benchmark")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print()

    all_results = []

    print("--- Transfer Size Sweep ---")
    size_results = run_size_sweep()
    all_results.extend(size_results)

    print("\n--- Transfer Count Sweep (fixed 64 MB total) ---")
    count_results = run_count_sweep(64)
    for r in count_results:
        r["sweep"] = "count"
    all_results.extend(count_results)

    print("\n--- D2H comparison (pinned, 64 MB) ---")
    d2h = bench_d2h_pinned(64 * 1024 * 1024, 3)
    all_results.append(d2h)

    print_summary(all_results)

    # Fit affine model from count sweep
    print("\n--- Affine Model Fit (H2D pinned, count sweep) ---")
    fit = fit_affine(count_results, "h2d_pinned")
    if "error" in fit:
        print(f"  Fit failed: {fit['error']}")
    else:
        print(f"  alpha (startup):     {fit['alpha_us']} us")
        print(f"  beta (per-transfer): {fit['beta_us']} us")
        print(f"  B (bandwidth):       {fit['B_MBps']} MB/s = {fit['B_GBps']} GB/s")
        print(f"  x* (min efficient):  {fit['x_star_KB']} KB")

    # Save
    output_dir = args.output_dir
    fit_path = os.path.join(output_dir, "timings", "fits", "affine_fit_pcie.json")
    os.makedirs(os.path.dirname(fit_path), exist_ok=True)
    with open(fit_path, "w") as f:
        json.dump(fit, f, indent=2)
    print(f"\nAffine fit written to {fit_path}")

    tsv_path = os.path.join(output_dir, "baseline", "results_pcie.tsv")
    keys = ["method", "transfer_size", "num_transfers", "total_bytes",
            "elapsed_ms", "bandwidth_MBps", "per_transfer_us"]
    with open(tsv_path, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in all_results:
            f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"Results written to {tsv_path}")


if __name__ == "__main__":
    main()
