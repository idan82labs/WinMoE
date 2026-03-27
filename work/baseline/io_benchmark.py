"""
Flash-MoE Windows V2 — Phase 0 I/O Benchmark Harness

Measures:
  1. SSD sequential read bandwidth at varying request sizes
  2. SSD random-offset read latency and bandwidth
  3. Explicit reads vs memory-mapped access comparison
  4. Cached vs uncached (FILE_FLAG_NO_BUFFERING) reads
  5. Multi-request overhead (varying request count at fixed total bytes)

Produces raw timing data suitable for affine model fitting:
  t(x, m) = alpha + beta*m + x/B

Usage:
  python io_benchmark.py --test-file <path> [--size-mb 256] [--output results.tsv]
"""

import argparse
import ctypes
import mmap
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

# Windows constants
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
GENERIC_READ = 0x80000000
OPEN_EXISTING = 3
FILE_SHARE_READ = 1


def create_test_file(path: str, size_mb: int) -> str:
    """Create a test file filled with random data to simulate expert weights."""
    size_bytes = size_mb * 1024 * 1024
    path = str(path)
    if os.path.exists(path) and os.path.getsize(path) == size_bytes:
        print(f"  Test file exists: {path} ({size_mb} MB)")
        return path

    print(f"  Creating test file: {path} ({size_mb} MB)...")
    chunk = 16 * 1024 * 1024  # 16 MB chunks
    rng = np.random.default_rng(42)
    with open(path, "wb") as f:
        written = 0
        while written < size_bytes:
            n = min(chunk, size_bytes - written)
            f.write(rng.bytes(n))
            written += n
    print(f"  Created: {path}")
    return path


def flush_os_cache():
    """Attempt to flush OS file cache. Requires admin on Windows."""
    try:
        # Try to flush file system buffers
        kernel32 = ctypes.windll.kernel32
        # SetSystemFileCacheSize with min=max=-1 purges standby list
        # This requires SeIncreaseQuotaPrivilege, so may fail without admin
        result = kernel32.SetSystemFileCacheSize(
            ctypes.c_size_t(-1), ctypes.c_size_t(-1), 0
        )
        if result:
            print("  OS file cache flushed successfully")
        else:
            print("  OS file cache flush failed (need admin?), using workaround")
            return False
        return True
    except Exception:
        return False


def evict_file_from_cache(path: str, size_bytes: int):
    """Read a large dummy buffer to push target file out of OS cache."""
    dummy_path = path + ".cache_evict_dummy"
    try:
        evict_size = min(size_bytes * 3, 512 * 1024 * 1024)
        rng = np.random.default_rng(99)
        with open(dummy_path, "wb") as f:
            written = 0
            chunk = 16 * 1024 * 1024
            while written < evict_size:
                n = min(chunk, evict_size - written)
                f.write(rng.bytes(n))
                written += n
        # Read it back to fill OS cache with other data
        with open(dummy_path, "rb") as f:
            while f.read(4 * 1024 * 1024):
                pass
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)


# ─── Test functions ───────────────────────────────────────────────────────────

def bench_explicit_read_sequential(path: str, request_size: int, total_bytes: int,
                                    use_no_buffering: bool = False) -> dict:
    """Explicit sequential reads using Python file I/O."""
    # Align request_size to sector boundary for NO_BUFFERING
    sector = 4096
    if use_no_buffering:
        request_size = max(sector, (request_size // sector) * sector)
        total_bytes = (total_bytes // request_size) * request_size

    num_requests = total_bytes // request_size
    if num_requests == 0:
        return {"error": "request_size > total_bytes"}

    if use_no_buffering:
        # Use ctypes for unbuffered reads on Windows
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.CreateFileW(
            path,
            GENERIC_READ,
            FILE_SHARE_READ,
            None,
            OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
            None,
        )
        if handle == -1:
            return {"error": f"CreateFileW failed: {ctypes.get_last_error()}"}

        buf = ctypes.create_string_buffer(request_size)
        bytes_read_total = 0
        t0 = time.perf_counter_ns()
        for _ in range(num_requests):
            bytes_read = ctypes.c_ulong(0)
            ok = kernel32.ReadFile(handle, buf, request_size, ctypes.byref(bytes_read), None)
            if not ok or bytes_read.value == 0:
                break
            bytes_read_total += bytes_read.value
        t1 = time.perf_counter_ns()
        kernel32.CloseHandle(handle)
    else:
        # Standard Python buffered read
        bytes_read_total = 0
        with open(path, "rb") as f:
            t0 = time.perf_counter_ns()
            for _ in range(num_requests):
                data = f.read(request_size)
                if not data:
                    break
                bytes_read_total += len(data)
            t1 = time.perf_counter_ns()

    elapsed_ns = t1 - t0
    elapsed_s = elapsed_ns / 1e9
    bw_mbps = (bytes_read_total / (1024 * 1024)) / elapsed_s if elapsed_s > 0 else 0

    return {
        "method": "explicit_unbuffered" if use_no_buffering else "explicit_buffered",
        "request_size": request_size,
        "num_requests": num_requests,
        "bytes_read": bytes_read_total,
        "elapsed_ns": elapsed_ns,
        "elapsed_ms": elapsed_ns / 1e6,
        "bandwidth_MBps": round(bw_mbps, 2),
        "per_request_us": round((elapsed_ns / num_requests) / 1000, 2),
    }


def bench_explicit_read_random(path: str, request_size: int, num_requests: int,
                                file_size: int, use_no_buffering: bool = False) -> dict:
    """Explicit random-offset reads."""
    sector = 4096
    if use_no_buffering:
        request_size = max(sector, (request_size // sector) * sector)

    rng = np.random.default_rng(7)
    max_offset = file_size - request_size
    if max_offset <= 0:
        return {"error": "file too small for request_size"}

    offsets = rng.integers(0, max_offset, size=num_requests)
    if use_no_buffering:
        offsets = (offsets // sector) * sector  # align

    bytes_read_total = 0

    if use_no_buffering:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.CreateFileW(
            path, GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING, None,
        )
        if handle == -1:
            return {"error": f"CreateFileW failed"}
        buf = ctypes.create_string_buffer(request_size)
        t0 = time.perf_counter_ns()
        for off in offsets:
            lo = int(off) & 0xFFFFFFFF
            hi = int(off) >> 32
            kernel32.SetFilePointer(handle, lo, ctypes.byref(ctypes.c_long(hi)), 0)
            br = ctypes.c_ulong(0)
            ok = kernel32.ReadFile(handle, buf, request_size, ctypes.byref(br), None)
            if ok:
                bytes_read_total += br.value
        t1 = time.perf_counter_ns()
        kernel32.CloseHandle(handle)
    else:
        with open(path, "rb") as f:
            t0 = time.perf_counter_ns()
            for off in offsets:
                f.seek(int(off))
                data = f.read(request_size)
                if data:
                    bytes_read_total += len(data)
            t1 = time.perf_counter_ns()

    elapsed_ns = t1 - t0
    elapsed_s = elapsed_ns / 1e9
    bw = (bytes_read_total / (1024 * 1024)) / elapsed_s if elapsed_s > 0 else 0

    return {
        "method": "random_unbuffered" if use_no_buffering else "random_buffered",
        "request_size": request_size,
        "num_requests": num_requests,
        "bytes_read": bytes_read_total,
        "elapsed_ns": elapsed_ns,
        "elapsed_ms": elapsed_ns / 1e6,
        "bandwidth_MBps": round(bw, 2),
        "per_request_us": round((elapsed_ns / num_requests) / 1000, 2),
    }


def bench_mmap_sequential(path: str, request_size: int, total_bytes: int) -> dict:
    """Memory-mapped sequential read."""
    file_size = os.path.getsize(path)
    total_bytes = min(total_bytes, file_size)
    num_requests = total_bytes // request_size

    bytes_read_total = 0
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        t0 = time.perf_counter_ns()
        for i in range(num_requests):
            offset = i * request_size
            chunk = mm[offset:offset + request_size]
            bytes_read_total += len(chunk)
        t1 = time.perf_counter_ns()
        mm.close()

    elapsed_ns = t1 - t0
    elapsed_s = elapsed_ns / 1e9
    bw = (bytes_read_total / (1024 * 1024)) / elapsed_s if elapsed_s > 0 else 0

    return {
        "method": "mmap_sequential",
        "request_size": request_size,
        "num_requests": num_requests,
        "bytes_read": bytes_read_total,
        "elapsed_ns": elapsed_ns,
        "elapsed_ms": elapsed_ns / 1e6,
        "bandwidth_MBps": round(bw, 2),
        "per_request_us": round((elapsed_ns / num_requests) / 1000, 2),
    }


def bench_mmap_random(path: str, request_size: int, num_requests: int,
                       file_size: int) -> dict:
    """Memory-mapped random-offset read."""
    rng = np.random.default_rng(7)
    max_offset = file_size - request_size
    if max_offset <= 0:
        return {"error": "file too small"}
    offsets = rng.integers(0, max_offset, size=num_requests)

    bytes_read_total = 0
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        t0 = time.perf_counter_ns()
        for off in offsets:
            chunk = mm[int(off):int(off) + request_size]
            bytes_read_total += len(chunk)
        t1 = time.perf_counter_ns()
        mm.close()

    elapsed_ns = t1 - t0
    elapsed_s = elapsed_ns / 1e9
    bw = (bytes_read_total / (1024 * 1024)) / elapsed_s if elapsed_s > 0 else 0

    return {
        "method": "mmap_random",
        "request_size": request_size,
        "num_requests": num_requests,
        "bytes_read": bytes_read_total,
        "elapsed_ns": elapsed_ns,
        "elapsed_ms": elapsed_ns / 1e6,
        "bandwidth_MBps": round(bw, 2),
        "per_request_us": round((elapsed_ns / num_requests) / 1000, 2),
    }


# ─── Request-size sweep ──────────────────────────────────────────────────────

def run_request_size_sweep(path: str, file_size: int, total_read_mb: int = 128):
    """Sweep request sizes from 4KB to 64MB for affine timing fit."""
    total_bytes = total_read_mb * 1024 * 1024
    sizes = [
        4 * 1024,       # 4 KB
        16 * 1024,      # 16 KB
        64 * 1024,      # 64 KB
        256 * 1024,     # 256 KB
        1024 * 1024,    # 1 MB
        4 * 1024 * 1024,    # 4 MB
        16 * 1024 * 1024,   # 16 MB
        64 * 1024 * 1024,   # 64 MB
    ]

    results = []
    for sz in sizes:
        if sz > file_size:
            continue
        adj_total = min(total_bytes, file_size)
        adj_total = (adj_total // sz) * sz
        if adj_total == 0:
            continue

        print(f"  Request size {sz // 1024} KB, total {adj_total // (1024*1024)} MB...")

        # Explicit buffered sequential
        r = bench_explicit_read_sequential(path, sz, adj_total, use_no_buffering=False)
        r["sweep"] = "request_size"
        results.append(r)

        # Explicit unbuffered sequential
        r = bench_explicit_read_sequential(path, sz, adj_total, use_no_buffering=True)
        r["sweep"] = "request_size"
        results.append(r)

        # mmap sequential
        r = bench_mmap_sequential(path, sz, adj_total)
        r["sweep"] = "request_size"
        results.append(r)

    return results


def run_request_count_sweep(path: str, file_size: int, fixed_total_mb: int = 64):
    """Fixed total bytes, varying request count to isolate per-request overhead."""
    total_bytes = fixed_total_mb * 1024 * 1024
    if total_bytes > file_size:
        total_bytes = file_size

    configs = [
        (total_bytes, 1),                    # 1 big read
        (total_bytes // 4, 4),               # 4 reads
        (total_bytes // 16, 16),             # 16 reads
        (total_bytes // 64, 64),             # 64 reads
        (total_bytes // 256, 256),           # 256 reads
        (total_bytes // 1024, 1024),         # 1024 reads
    ]

    results = []
    for req_size, num_req in configs:
        if req_size < 4096:
            continue
        req_size = (req_size // 4096) * 4096  # align
        actual_total = req_size * num_req
        print(f"  {num_req} requests x {req_size // 1024} KB = {actual_total // (1024*1024)} MB...")

        r = bench_explicit_read_sequential(path, req_size, actual_total, use_no_buffering=True)
        r["sweep"] = "request_count"
        results.append(r)

        r = bench_explicit_read_sequential(path, req_size, actual_total, use_no_buffering=False)
        r["sweep"] = "request_count"
        results.append(r)

    return results


def run_random_read_sweep(path: str, file_size: int):
    """Random reads at various sizes to measure random access overhead."""
    sizes = [4096, 64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024]
    num_requests = 100

    results = []
    for sz in sizes:
        if sz > file_size // 2:
            continue
        print(f"  Random {sz // 1024} KB x {num_requests}...")

        r = bench_explicit_read_random(path, sz, num_requests, file_size, use_no_buffering=True)
        r["sweep"] = "random"
        results.append(r)

        r = bench_explicit_read_random(path, sz, num_requests, file_size, use_no_buffering=False)
        r["sweep"] = "random"
        results.append(r)

        r = bench_mmap_random(path, sz, num_requests, file_size)
        r["sweep"] = "random"
        results.append(r)

    return results


# ─── Affine fit ───────────────────────────────────────────────────────────────

def fit_affine_model(results: list) -> dict:
    """Fit t(x, m) = alpha + beta*m + x/B from request-size sweep data.

    Uses the request_count sweep results (fixed total, varying m).
    """
    # Filter to unbuffered explicit reads from request_count sweep
    pts = [r for r in results
           if r.get("sweep") == "request_count"
           and r.get("method") == "explicit_unbuffered"
           and "error" not in r]

    if len(pts) < 3:
        return {"error": "not enough data points for fit"}

    # Build system: t = alpha + beta*m + x/B
    # Rewrite as: t = alpha + beta*m + (1/B)*x
    # Variables: [alpha, beta, 1/B]
    A = []
    b = []
    for p in pts:
        m = p["num_requests"]
        x = p["bytes_read"]
        t = p["elapsed_ns"] / 1e9  # seconds
        A.append([1.0, float(m), float(x)])
        b.append(t)

    A = np.array(A)
    b_vec = np.array(b)

    # Least squares fit
    result, residuals, rank, sv = np.linalg.lstsq(A, b_vec, rcond=None)
    alpha, beta, inv_B = result

    B = 1.0 / inv_B if abs(inv_B) > 1e-15 else float("inf")
    x_star = alpha * B if B != float("inf") else float("nan")

    return {
        "alpha_s": round(alpha, 9),
        "alpha_us": round(alpha * 1e6, 2),
        "beta_s": round(beta, 9),
        "beta_us": round(beta * 1e6, 2),
        "B_bytes_per_s": round(B, 0),
        "B_MBps": round(B / (1024 * 1024), 1),
        "B_GBps": round(B / (1024 * 1024 * 1024), 3),
        "x_star_bytes": round(x_star, 0) if not np.isnan(x_star) else "N/A",
        "x_star_KB": round(x_star / 1024, 1) if not np.isnan(x_star) else "N/A",
        "num_points": len(pts),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def write_results_tsv(results: list, output_path: str):
    """Write results as TSV."""
    if not results:
        return
    keys = ["sweep", "method", "request_size", "num_requests", "bytes_read",
            "elapsed_ms", "bandwidth_MBps", "per_request_us"]
    with open(output_path, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in results:
            if "error" in r:
                continue
            row = [str(r.get(k, "")) for k in keys]
            f.write("\t".join(row) + "\n")
    print(f"\nResults written to {output_path}")


def print_summary(results: list):
    """Print a readable summary table."""
    print("\n" + "=" * 90)
    print(f"{'Method':<25} {'ReqSize':>10} {'NumReq':>8} {'MB':>8} {'ms':>10} {'MB/s':>10} {'us/req':>10}")
    print("-" * 90)
    for r in results:
        if "error" in r:
            continue
        method = r.get("method", "?")
        rsz = r.get("request_size", 0)
        rsz_str = f"{rsz // 1024}K" if rsz < 1024 * 1024 else f"{rsz // (1024*1024)}M"
        print(f"{method:<25} {rsz_str:>10} {r.get('num_requests', 0):>8} "
              f"{r.get('bytes_read', 0) / (1024*1024):>8.1f} "
              f"{r.get('elapsed_ms', 0):>10.1f} {r.get('bandwidth_MBps', 0):>10.1f} "
              f"{r.get('per_request_us', 0):>10.1f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Flash-MoE I/O Benchmark")
    parser.add_argument("--test-file", type=str,
                        help="Path to test file (will be created if missing)")
    parser.add_argument("--size-mb", type=int, default=256,
                        help="Test file size in MB (default 256)")
    parser.add_argument("--output", type=str, default="results.tsv",
                        help="Output TSV path")
    parser.add_argument("--sweep-total-mb", type=int, default=128,
                        help="Total MB to read in request-size sweep")
    parser.add_argument("--skip-random", action="store_true",
                        help="Skip random read tests")
    parser.add_argument("--drive", type=str, default=None,
                        help="Drive letter to test (e.g., D:)")
    args = parser.parse_args()

    # Determine test file location
    if args.test_file:
        test_path = args.test_file
    elif args.drive:
        test_path = os.path.join(args.drive + "\\", "flash_moe_io_test.bin")
    else:
        # Default: use the work directory
        test_path = os.path.join(os.path.dirname(__file__), "raw", "io_test.bin")

    os.makedirs(os.path.dirname(test_path) or ".", exist_ok=True)

    print("=" * 60)
    print("Flash-MoE Windows V2 — I/O Benchmark")
    print("=" * 60)
    print(f"Test file: {test_path}")
    print(f"File size: {args.size_mb} MB")
    print()

    # Create test file
    test_path = create_test_file(test_path, args.size_mb)
    file_size = os.path.getsize(test_path)

    all_results = []

    # 1. Request-size sweep (sequential)
    print("\n--- Request-Size Sweep (Sequential) ---")
    results = run_request_size_sweep(test_path, file_size, args.sweep_total_mb)
    all_results.extend(results)

    # 2. Request-count sweep (fixed total, varying m)
    print("\n--- Request-Count Sweep (Fixed Total) ---")
    results = run_request_count_sweep(test_path, file_size)
    all_results.extend(results)

    # 3. Random read sweep
    if not args.skip_random:
        print("\n--- Random Read Sweep ---")
        results = run_random_read_sweep(test_path, file_size)
        all_results.extend(results)

    # Print summary
    print_summary(all_results)

    # Fit affine model
    print("\n--- Affine Timing Model Fit ---")
    fit = fit_affine_model(all_results)
    if "error" in fit:
        print(f"  Fit failed: {fit['error']}")
    else:
        print(f"  alpha (startup):     {fit['alpha_us']} us")
        print(f"  beta (per-request):  {fit['beta_us']} us")
        print(f"  B (bandwidth):       {fit['B_MBps']} MB/s = {fit['B_GBps']} GB/s")
        print(f"  x* (min efficient):  {fit['x_star_KB']} KB")
        print(f"  Data points used:    {fit['num_points']}")

    # Write TSV
    output_path = os.path.join(os.path.dirname(test_path), "..", args.output)
    write_results_tsv(all_results, output_path)

    # Write affine fit
    fit_path = os.path.join(os.path.dirname(test_path), "..", "affine_fit_ssd.json")
    import json
    with open(fit_path, "w") as f:
        json.dump(fit, f, indent=2)
    print(f"Affine fit written to {fit_path}")

    # Cleanup
    print(f"\nNote: Test file at {test_path} can be deleted after benchmarking.")
    return all_results, fit


if __name__ == "__main__":
    main()
