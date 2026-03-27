"""
Flash-MoE Windows V2 — Trace-Driven Cache Simulator

Simulates tiered expert caching (VRAM / RAM / SSD) under multiple policies
and computes miss-service demand distributions.

Inputs:
  - Routing traces: per-token, per-layer expert activation IDs
  - Timing parameters: affine fits for SSD, PCIe, dequant stages
  - Cache capacities: VRAM and RAM tier sizes (in number of expert blocks)

Outputs:
  - Hit/miss rates per tier
  - SSD bytes and request counts
  - PCIe bytes and transfer counts
  - Service-demand distribution S_t(C)
  - Load factor rho(C) = E[S_t] / T_c
  - Policy gap comparisons (static vs recency vs oracle)

Policies:
  - static_lfu: pin most popular experts based on global frequency
  - lru: standard LRU eviction
  - arc: Adaptive Replacement Cache
  - hybrid: static head + LRU tail in VRAM, static RAM
  - oracle: offline optimal (Belady's MIN)

Usage:
  python moe_cache_simulator.py --trace traces.npz --config config.json
"""

import argparse
import json
import os
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TimingParams:
    """Affine timing model: t(x, m) = alpha + beta*m + x/B"""
    # SSD
    ssd_alpha_s: float = 0.0        # SSD startup (seconds)
    ssd_beta_s: float = 42e-6       # SSD per-request overhead (seconds)
    ssd_B_bytes_per_s: float = 2.118e9  # SSD bandwidth (bytes/s)
    # PCIe
    pcie_alpha_s: float = 0.0       # PCIe startup
    pcie_beta_s: float = 11e-6      # PCIe per-transfer overhead
    pcie_B_bytes_per_s: float = 19e9  # PCIe bandwidth (bytes/s, pinned)
    # Compute
    T_c_s: float = 0.005            # compute time per layer (seconds) — placeholder

    def ssd_time(self, bytes_: float, num_requests: int = 1) -> float:
        return self.ssd_alpha_s + self.ssd_beta_s * num_requests + bytes_ / self.ssd_B_bytes_per_s

    def pcie_time(self, bytes_: float, num_transfers: int = 1) -> float:
        return self.pcie_alpha_s + self.pcie_beta_s * num_transfers + bytes_ / self.pcie_B_bytes_per_s


@dataclass
class ModelConfig:
    """MoE model configuration."""
    num_layers: int = 60
    num_experts: int = 512
    num_active: int = 10      # K (routed)
    has_shared: bool = True
    expert_size_bytes: int = 6_291_456  # ~6.29 MB at Q4 for Qwen3.5-397B
    shared_expert_size_bytes: int = 6_291_456

    @property
    def total_experts(self) -> int:
        return self.num_layers * self.num_experts

    @property
    def per_layer_miss_bytes(self) -> float:
        """Full per-layer miss payload (all K experts + shared from SSD)."""
        return self.num_active * self.expert_size_bytes + (
            self.shared_expert_size_bytes if self.has_shared else 0
        )


@dataclass
class SimResult:
    """Results from a single simulation run."""
    policy: str
    vram_capacity: int  # number of expert blocks in VRAM
    ram_capacity: int   # number of expert blocks in RAM
    total_tokens: int = 0
    total_layers: int = 0

    # Counters
    vram_hits: int = 0
    ram_hits: int = 0
    ssd_hits: int = 0   # = misses from cache
    ssd_bytes: float = 0.0
    ssd_requests: int = 0
    pcie_bytes: float = 0.0
    pcie_transfers: int = 0

    # Per-layer service demand (seconds)
    service_demands: list = field(default_factory=list)

    @property
    def total_accesses(self) -> int:
        return self.vram_hits + self.ram_hits + self.ssd_hits

    @property
    def vram_hit_rate(self) -> float:
        return self.vram_hits / max(1, self.total_accesses)

    @property
    def ram_hit_rate(self) -> float:
        return self.ram_hits / max(1, self.total_accesses)

    @property
    def ssd_miss_rate(self) -> float:
        return self.ssd_hits / max(1, self.total_accesses)

    def summary(self) -> dict:
        sd = np.array(self.service_demands) if self.service_demands else np.array([0.0])
        return {
            "policy": self.policy,
            "vram_cap": self.vram_capacity,
            "ram_cap": self.ram_capacity,
            "tokens": self.total_tokens,
            "vram_hit%": round(self.vram_hit_rate * 100, 2),
            "ram_hit%": round(self.ram_hit_rate * 100, 2),
            "ssd_miss%": round(self.ssd_miss_rate * 100, 2),
            "ssd_GB": round(self.ssd_bytes / 1e9, 3),
            "ssd_reqs": self.ssd_requests,
            "pcie_GB": round(self.pcie_bytes / 1e9, 3),
            "mean_service_ms": round(np.mean(sd) * 1000, 3),
            "p95_service_ms": round(np.percentile(sd, 95) * 1000, 3),
            "p99_service_ms": round(np.percentile(sd, 99) * 1000, 3),
            "max_service_ms": round(np.max(sd) * 1000, 3),
        }


# ─── Cache Policies ──────────────────────────────────────────────────────────

class StaticLFUCache:
    """Static hotset: precompute top-C experts by global frequency."""

    def __init__(self, capacity: int, expert_freqs: dict):
        self.capacity = capacity
        # Sort by frequency descending
        sorted_experts = sorted(expert_freqs.items(), key=lambda x: -x[1])
        self.pinned = set(eid for eid, _ in sorted_experts[:capacity])

    def contains(self, expert_id: int) -> bool:
        return expert_id in self.pinned

    def access(self, expert_id: int):
        pass  # static, no updates


class LRUCache:
    """Standard LRU cache using OrderedDict."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def contains(self, expert_id: int) -> bool:
        return expert_id in self.cache

    def access(self, expert_id: int):
        if self.capacity <= 0:
            return
        if expert_id in self.cache:
            self.cache.move_to_end(expert_id)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)  # evict LRU
            self.cache[expert_id] = True


class HybridCache:
    """Static head + LRU tail."""

    def __init__(self, static_capacity: int, adaptive_capacity: int, expert_freqs: dict):
        self.static = StaticLFUCache(static_capacity, expert_freqs)
        self.adaptive = LRUCache(adaptive_capacity)

    def contains(self, expert_id: int) -> bool:
        return self.static.contains(expert_id) or self.adaptive.contains(expert_id)

    def access(self, expert_id: int):
        if not self.static.contains(expert_id):
            self.adaptive.access(expert_id)


class OracleCache:
    """Offline optimal (Belady's MIN): evict the expert whose next use is farthest."""

    def __init__(self, capacity: int, full_trace: list):
        self.capacity = capacity
        self.cache = set()
        # Precompute next-use index for each (position, expert_id)
        self.next_use = {}
        last_seen = {}
        # Process trace in reverse
        for i in range(len(full_trace) - 1, -1, -1):
            eid = full_trace[i]
            if eid in last_seen:
                self.next_use[i] = last_seen[eid]
            else:
                self.next_use[i] = float("inf")
            last_seen[eid] = i
        self._pos = 0

    def contains(self, expert_id: int) -> bool:
        return expert_id in self.cache

    def access(self, expert_id: int):
        if expert_id in self.cache:
            self._pos += 1
            return
        if len(self.cache) >= self.capacity:
            # Evict the one with farthest next use
            farthest = -1
            evict = None
            for eid in self.cache:
                # Find this expert's next use after current position
                # Simplified: just check next_use from last access
                nu = self.next_use.get(self._pos, float("inf"))
                # Actually we need per-expert next use
                pass
            # Simpler: scan forward from _pos for each cached expert
            farthest_use = -1
            evict_eid = None
            for eid in self.cache:
                nu = float("inf")
                # This is O(n) per eviction — acceptable for simulation
                for j in range(self._pos + 1, len(self._all_trace)):
                    if self._all_trace[j] == eid:
                        nu = j
                        break
                if nu > farthest_use:
                    farthest_use = nu
                    evict_eid = eid
            if evict_eid is not None:
                self.cache.discard(evict_eid)
        self.cache.add(expert_id)
        self._pos += 1


class EfficientOracleCache:
    """Efficient Belady's MIN using precomputed next-use chains."""

    def __init__(self, capacity: int, flat_trace: list):
        self.capacity = capacity
        self.cache = set()
        n = len(flat_trace)

        # Build next-use array: for each position i, next_use[i] = next index where flat_trace[i] appears
        self.next_use_arr = np.full(n, n + 1, dtype=np.int64)
        last_seen = {}
        for i in range(n - 1, -1, -1):
            eid = flat_trace[i]
            if eid in last_seen:
                self.next_use_arr[i] = last_seen[eid]
            last_seen[eid] = i

        self.flat_trace = flat_trace
        # For each expert currently in cache, track its current "next use" index
        self.expert_next = {}  # expert_id -> next use index
        self._pos = 0

    def contains(self, expert_id: int) -> bool:
        return expert_id in self.cache

    def access_at(self, pos: int, expert_id: int):
        """Access expert at trace position pos."""
        if expert_id in self.cache:
            # Update next use
            self.expert_next[expert_id] = self.next_use_arr[pos]
            return
        if len(self.cache) >= self.capacity and self.capacity > 0:
            # Evict expert with farthest next use
            evict_eid = max(self.cache, key=lambda e: self.expert_next.get(e, len(self.flat_trace) + 1))
            self.cache.discard(evict_eid)
            del self.expert_next[evict_eid]
        self.cache.add(expert_id)
        self.expert_next[expert_id] = self.next_use_arr[pos]


# ─── Trace generation ─────────────────────────────────────────────────────────

def generate_synthetic_trace(
    num_tokens: int,
    model_config: ModelConfig,
    zipf_s: float = 1.2,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic routing traces with Zipf-distributed expert popularity.

    Returns: array of shape (num_tokens, num_layers, K) with expert IDs.
    """
    rng = np.random.default_rng(seed)

    # Per-layer expert popularity (Zipf distribution)
    expert_ids = np.arange(model_config.num_experts)
    ranks = np.arange(1, model_config.num_experts + 1, dtype=np.float64)
    weights = 1.0 / (ranks ** zipf_s)
    probs = weights / weights.sum()

    traces = np.zeros(
        (num_tokens, model_config.num_layers, model_config.num_active),
        dtype=np.int32,
    )

    for layer in range(model_config.num_layers):
        # Shuffle expert ordering per layer (different popularity per layer)
        layer_perm = rng.permutation(model_config.num_experts)
        layer_probs = probs.copy()

        for token in range(num_tokens):
            # Add some token-level noise to simulate non-stationarity
            noisy_probs = layer_probs * (1.0 + 0.1 * rng.standard_normal(model_config.num_experts))
            noisy_probs = np.maximum(noisy_probs, 0)
            noisy_probs /= noisy_probs.sum()

            # Select top-K by sampling without replacement
            chosen = rng.choice(
                model_config.num_experts,
                size=model_config.num_active,
                replace=False,
                p=noisy_probs,
            )
            traces[token, layer] = layer_perm[chosen]

    return traces


def compute_expert_frequencies(traces: np.ndarray, num_layers: int, num_experts: int) -> dict:
    """Compute global expert frequencies from traces.

    Returns: dict mapping (layer, expert_id) -> frequency count.
    """
    freqs = defaultdict(int)
    num_tokens = traces.shape[0]
    for token in range(num_tokens):
        for layer in range(num_layers):
            for expert in traces[token, layer]:
                freqs[(layer, int(expert))] += 1
    return dict(freqs)


# ─── Simulator core ───────────────────────────────────────────────────────────

def simulate(
    traces: np.ndarray,
    model_config: ModelConfig,
    timing: TimingParams,
    vram_capacity: int,
    ram_capacity: int,
    policy: str = "static_lfu",
    expert_freqs: Optional[dict] = None,
) -> SimResult:
    """Run cache simulation.

    Args:
        traces: (num_tokens, num_layers, K) expert activation IDs
        model_config: model architecture config
        timing: calibrated timing parameters
        vram_capacity: number of expert blocks in VRAM cache
        ram_capacity: number of expert blocks in RAM cache
        policy: "static_lfu", "lru", "hybrid", "oracle"
        expert_freqs: precomputed frequency dict (required for static policies)

    Returns:
        SimResult with all counters and service demands
    """
    num_tokens, num_layers, K = traces.shape
    result = SimResult(
        policy=policy,
        vram_capacity=vram_capacity,
        ram_capacity=ram_capacity,
        total_tokens=num_tokens,
        total_layers=num_tokens * num_layers,
    )

    if expert_freqs is None:
        expert_freqs = compute_expert_frequencies(traces, num_layers, model_config.num_experts)

    # Initialize caches based on policy
    if policy == "static_lfu":
        vram_cache = StaticLFUCache(vram_capacity, expert_freqs)
        ram_cache = StaticLFUCache(ram_capacity, {
            k: v for k, v in expert_freqs.items() if not vram_cache.contains(k)
        })
    elif policy == "lru":
        vram_cache = LRUCache(vram_capacity)
        ram_cache = LRUCache(ram_capacity)
    elif policy == "hybrid":
        # 70% static, 30% adaptive in VRAM; fully static RAM
        static_vram = int(vram_capacity * 0.7)
        adaptive_vram = vram_capacity - static_vram
        vram_cache = HybridCache(static_vram, adaptive_vram, expert_freqs)
        ram_cache = StaticLFUCache(ram_capacity, {
            k: v for k, v in expert_freqs.items()
            if not (isinstance(vram_cache, HybridCache) and vram_cache.contains(k))
        })
    elif policy == "oracle":
        # Flatten trace for oracle
        flat_trace = []
        for t in range(num_tokens):
            for l in range(num_layers):
                for e in traces[t, l]:
                    flat_trace.append((l, int(e)))
        oracle_vram = EfficientOracleCache(vram_capacity, flat_trace)
        oracle_ram = EfficientOracleCache(ram_capacity, [])  # simplified
        # Oracle uses combined VRAM+RAM capacity
        combined_oracle = EfficientOracleCache(vram_capacity + ram_capacity, flat_trace)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    expert_size = model_config.expert_size_bytes

    # Simulate token by token
    flat_pos = 0
    for token in range(num_tokens):
        for layer in range(num_layers):
            layer_ssd_bytes = 0.0
            layer_ssd_reqs = 0
            layer_pcie_bytes = 0.0
            layer_pcie_transfers = 0

            for k_idx in range(K):
                expert_id_raw = int(traces[token, layer, k_idx])
                expert_key = (layer, expert_id_raw)

                if policy == "oracle":
                    if combined_oracle.contains(expert_key):
                        if flat_pos < len(flat_trace):
                            combined_oracle.access_at(flat_pos, expert_key)
                        # Decide if it's "VRAM" or "RAM" — for oracle, treat first
                        # vram_capacity slots as VRAM hits, rest as RAM hits
                        result.vram_hits += 1  # simplified
                    else:
                        if flat_pos < len(flat_trace):
                            combined_oracle.access_at(flat_pos, expert_key)
                        result.ssd_hits += 1
                        layer_ssd_bytes += expert_size
                        layer_ssd_reqs += 1
                        layer_pcie_bytes += expert_size
                        layer_pcie_transfers += 1
                else:
                    if vram_cache.contains(expert_key):
                        result.vram_hits += 1
                        vram_cache.access(expert_key) if hasattr(vram_cache, 'access') else None
                    elif ram_cache.contains(expert_key):
                        result.ram_hits += 1
                        ram_cache.access(expert_key) if hasattr(ram_cache, 'access') else None
                        # RAM hit still needs PCIe transfer to GPU
                        layer_pcie_bytes += expert_size
                        layer_pcie_transfers += 1
                    else:
                        result.ssd_hits += 1
                        layer_ssd_bytes += expert_size
                        layer_ssd_reqs += 1
                        layer_pcie_bytes += expert_size
                        layer_pcie_transfers += 1
                        # Update adaptive caches
                        if policy == "lru":
                            # Promotion: SSD miss loads into RAM, RAM eviction may go to VRAM
                            ram_cache.access(expert_key)
                        elif policy == "hybrid":
                            vram_cache.access(expert_key)

                flat_pos += 1

            # Compute per-layer service demand (causal model)
            ssd_time = timing.ssd_time(layer_ssd_bytes, layer_ssd_reqs) if layer_ssd_reqs > 0 else 0
            pcie_time = timing.pcie_time(layer_pcie_bytes, layer_pcie_transfers) if layer_pcie_transfers > 0 else 0
            service_demand = ssd_time + pcie_time

            result.service_demands.append(service_demand)
            result.ssd_bytes += layer_ssd_bytes
            result.ssd_requests += layer_ssd_reqs
            result.pcie_bytes += layer_pcie_bytes
            result.pcie_transfers += layer_pcie_transfers

    return result


# ─── Analysis functions ───────────────────────────────────────────────────────

def compute_cache_curve(
    traces: np.ndarray,
    model_config: ModelConfig,
    timing: TimingParams,
    policy: str,
    ram_capacity: int,
    vram_capacities: list,
    expert_freqs: dict,
) -> list:
    """Compute service demand vs VRAM capacity curve."""
    results = []
    for vc in vram_capacities:
        print(f"  {policy} VRAM={vc} RAM={ram_capacity}...", end="", flush=True)
        r = simulate(traces, model_config, timing, vc, ram_capacity, policy, expert_freqs)
        s = r.summary()
        print(f" miss={s['ssd_miss%']}% mean_svc={s['mean_service_ms']}ms")
        results.append(s)
    return results


def compute_load_factor(service_demands: list, T_c: float) -> float:
    """rho(C) = E[S_t(C)] / T_c"""
    return np.mean(service_demands) / T_c


def find_critical_cache_size(results: list, T_c_ms: float, quantile: float = 0.95) -> Optional[dict]:
    """Find smallest cache where quantile service demand < compute time."""
    for r in results:
        if r[f"p{int(quantile*100)}_service_ms"] <= T_c_ms:
            return r
    return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MoE Cache Simulator")
    parser.add_argument("--trace", type=str, help="Path to trace file (.npz)")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--num-tokens", type=int, default=200,
                        help="Number of tokens for synthetic trace")
    parser.add_argument("--zipf-s", type=float, default=1.2,
                        help="Zipf exponent for synthetic trace")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--policies", type=str, default="static_lfu,lru,hybrid",
                        help="Comma-separated policies to test")
    args = parser.parse_args()

    # Model config
    model = ModelConfig()

    # Timing
    timing = TimingParams()

    # Load or generate traces
    if args.trace and os.path.exists(args.trace):
        data = np.load(args.trace)
        traces = data["traces"]
        print(f"Loaded traces: {traces.shape}")
    else:
        print(f"Generating synthetic trace ({args.num_tokens} tokens, zipf_s={args.zipf_s})...")
        traces = generate_synthetic_trace(args.num_tokens, model, args.zipf_s)
        print(f"Trace shape: {traces.shape}")

        # Save synthetic trace
        os.makedirs(args.output_dir, exist_ok=True)
        trace_path = os.path.join(args.output_dir, "synthetic_trace.npz")
        np.savez_compressed(trace_path, traces=traces)
        print(f"Saved to {trace_path}")

    num_tokens, num_layers, K = traces.shape
    print(f"Tokens: {num_tokens}, Layers: {num_layers}, K: {K}")

    # Compute frequencies
    print("Computing expert frequencies...")
    freqs = compute_expert_frequencies(traces, num_layers, model.num_experts)
    print(f"Unique (layer, expert) pairs: {len(freqs)}")

    # Total expert blocks in the system
    total_blocks = num_layers * model.num_experts
    print(f"Total expert blocks: {total_blocks}")

    # Define capacity sweep
    # VRAM: 8 GB = ~1,270 expert blocks at 6.29 MB each
    # Usable VRAM after model overhead: maybe 4-6 GB = ~636-953 blocks
    # RAM: 32 GB usable = ~5,087 blocks
    vram_caps = [0, 100, 200, 500, 1000, 1500, 2000, 3000]
    ram_cap = 5000  # ~31.5 GB of RAM dedicated to expert cache

    policies = args.policies.split(",")
    all_results = {}

    for policy in policies:
        if policy == "oracle" and num_tokens > 50:
            print(f"\nSkipping oracle for {num_tokens} tokens (too slow)")
            continue
        print(f"\n{'='*60}")
        print(f"Policy: {policy}")
        print(f"{'='*60}")
        curve = compute_cache_curve(
            traces, model, timing, policy, ram_cap, vram_caps, freqs,
        )
        all_results[policy] = curve

    # Print comparison table
    print(f"\n{'='*100}")
    print(f"{'Policy':<15} {'VRAM':<8} {'RAM':<8} {'VRAM%':>8} {'RAM%':>8} {'SSD%':>8} "
          f"{'MeanSvc':>10} {'P95Svc':>10} {'P99Svc':>10}")
    print("-" * 100)
    for policy, curve in all_results.items():
        for r in curve:
            print(f"{r['policy']:<15} {r['vram_cap']:<8} {r['ram_cap']:<8} "
                  f"{r['vram_hit%']:>7.1f}% {r['ram_hit%']:>7.1f}% {r['ssd_miss%']:>7.1f}% "
                  f"{r['mean_service_ms']:>9.2f}ms {r['p95_service_ms']:>9.2f}ms "
                  f"{r['p99_service_ms']:>9.2f}ms")
    print("=" * 100)

    # Compute load factors
    T_c_ms = timing.T_c_s * 1000
    print(f"\nCompute time T_c = {T_c_ms} ms")
    for policy, curve in all_results.items():
        for r in curve:
            rho = r["mean_service_ms"] / T_c_ms
            status = "VIABLE" if rho < 1.0 else "NOT VIABLE"
            if r["vram_cap"] in [0, 500, 1000, 2000, 3000]:
                print(f"  {policy} VRAM={r['vram_cap']}: rho={rho:.3f} [{status}]")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "sim_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
