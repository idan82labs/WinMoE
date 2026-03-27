"""
Trace Replay Harness — Workstream 1

Replays routing traces through a configurable cache simulator with
calibrated timing model. This is the benchmark foundation for all
engine workstreams.

Inputs:
  - Routing traces (expert IDs per token per layer)
  - Timing calibration (SSD/PCIe/compute affine models)
  - Cache policy (static LFU, LRU, or oracle)
  - Cache sizes (VRAM blocks, RAM blocks)

Outputs:
  - Simulated tok/s
  - Cache hit rates per tier
  - SSD miss rate
  - Service demand distribution
"""
import json
import os
import sys
import time
from collections import defaultdict, OrderedDict
import numpy as np


class TieredCache:
    """Three-tier cache: VRAM (hot) -> RAM (warm) -> SSD (cold)."""

    def __init__(self, vram_capacity, ram_capacity, policy="lfu"):
        self.vram_capacity = vram_capacity
        self.ram_capacity = ram_capacity
        self.policy = policy

        # VRAM: static LFU hotset (pinned at init, never evicted during run)
        self.vram_set = set()

        # RAM: LRU or LFU cache
        self.ram_cache = OrderedDict()  # key -> access_count
        self.ram_freq = defaultdict(int)  # for LFU

        # Stats
        self.stats = {"vram_hits": 0, "ram_hits": 0, "ssd_misses": 0, "total": 0}

    def init_vram_hotset(self, frequency_map, num_layers, experts_per_layer):
        """Pin the top-N most frequent expert blocks in VRAM."""
        # frequency_map: {(layer, expert_id): count}
        sorted_experts = sorted(frequency_map.items(), key=lambda x: -x[1])
        for (key, count) in sorted_experts[:self.vram_capacity]:
            self.vram_set.add(key)

    def access(self, layer, expert_id):
        """Access an expert block. Returns tier: 'vram', 'ram', or 'ssd'."""
        key = (layer, expert_id)
        self.stats["total"] += 1

        # Check VRAM (static hotset)
        if key in self.vram_set:
            self.stats["vram_hits"] += 1
            return "vram"

        # Check RAM cache
        if key in self.ram_cache:
            self.stats["ram_hits"] += 1
            # Move to end (most recently used)
            self.ram_cache.move_to_end(key)
            self.ram_freq[key] += 1
            return "ram"

        # SSD miss — load into RAM cache
        self.stats["ssd_misses"] += 1
        self._insert_ram(key)
        return "ssd"

    def _insert_ram(self, key):
        """Insert into RAM cache, evicting if necessary."""
        if len(self.ram_cache) >= self.ram_capacity:
            if self.policy == "lru":
                evicted, _ = self.ram_cache.popitem(last=False)
                del self.ram_freq[evicted]
            elif self.policy == "lfu":
                # Evict least frequently used
                min_key = min(self.ram_cache, key=lambda k: self.ram_freq[k])
                del self.ram_cache[min_key]
                del self.ram_freq[min_key]
        self.ram_cache[key] = 1
        self.ram_freq[key] = 1

    def get_stats(self):
        total = self.stats["total"]
        if total == 0:
            return self.stats
        return {
            **self.stats,
            "vram_hit_rate": self.stats["vram_hits"] / total,
            "ram_hit_rate": self.stats["ram_hits"] / total,
            "ssd_miss_rate": self.stats["ssd_misses"] / total,
            "combined_cache_hit_rate": (self.stats["vram_hits"] + self.stats["ram_hits"]) / total,
        }


class TimingModel:
    """Calibrated affine timing model for each pipeline stage."""

    def __init__(self, expert_size_bytes, ssd_bw=2.1e9, ssd_overhead=42e-6,
                 pcie_bw=19e9, pcie_overhead=11e-6, compute_ms_per_layer=7.12):
        self.expert_size = expert_size_bytes
        self.ssd_bw = ssd_bw
        self.ssd_overhead = ssd_overhead
        self.pcie_bw = pcie_bw
        self.pcie_overhead = pcie_overhead
        self.compute_ms = compute_ms_per_layer  # from Stage 0

    def layer_time_ms(self, k_active, ssd_miss_count, ram_hit_count, vram_hit_count):
        """Compute per-layer time given cache hit/miss counts."""
        # SSD reads (cold misses)
        ssd_bytes = ssd_miss_count * self.expert_size
        ssd_ms = (ssd_bytes / self.ssd_bw + ssd_miss_count * self.ssd_overhead) * 1000

        # RAM->GPU transfers (warm hits still need PCIe)
        ram_bytes = ram_hit_count * self.expert_size
        pcie_ms = (ram_bytes / self.pcie_bw + ram_hit_count * self.pcie_overhead) * 1000

        # VRAM hits: no transfer, just compute
        # But we still pay compute cost for all K experts
        compute = self.compute_ms

        # Causal model: SSD + PCIe + compute (conservative, no overlap)
        total_causal = ssd_ms + pcie_ms + compute

        # Overlap model: max(SSD+PCIe, compute) — optimistic, assumes pipeline
        total_overlap = max(ssd_ms + pcie_ms, compute)

        return {
            "ssd_ms": ssd_ms,
            "pcie_ms": pcie_ms,
            "compute_ms": compute,
            "total_causal_ms": total_causal,
            "total_overlap_ms": total_overlap,
        }


def replay_traces(traces, K, num_layers, num_experts, cache, timing_model):
    """Replay routing traces through cache and timing model."""
    num_tokens = traces.shape[0]
    layer_times_causal = []
    layer_times_overlap = []

    for t in range(num_tokens):
        token_causal = 0
        token_overlap = 0
        for l in range(num_layers):
            experts = traces[t, l, :K].tolist()

            ssd = 0
            ram = 0
            vram = 0
            for e in experts:
                tier = cache.access(l, e)
                if tier == "vram":
                    vram += 1
                elif tier == "ram":
                    ram += 1
                else:
                    ssd += 1

            timing = timing_model.layer_time_ms(K, ssd, ram, vram)
            token_causal += timing["total_causal_ms"]
            token_overlap += timing["total_overlap_ms"]

        layer_times_causal.append(token_causal)
        layer_times_overlap.append(token_overlap)

    return {
        "num_tokens": num_tokens,
        "mean_causal_ms": np.mean(layer_times_causal),
        "mean_overlap_ms": np.mean(layer_times_overlap),
        "p95_causal_ms": np.percentile(layer_times_causal, 95),
        "p95_overlap_ms": np.percentile(layer_times_overlap, 95),
        "causal_tps": 1000.0 / np.mean(layer_times_causal) if np.mean(layer_times_causal) > 0 else 0,
        "overlap_tps": 1000.0 / np.mean(layer_times_overlap) if np.mean(layer_times_overlap) > 0 else 0,
        "cache_stats": cache.get_stats(),
    }


def build_frequency_map(traces, K, num_layers):
    """Build global expert frequency map from traces."""
    freq = defaultdict(int)
    for t in range(traces.shape[0]):
        for l in range(num_layers):
            for k in range(K):
                freq[(l, int(traces[t, l, k]))] += 1
    return freq


def main():
    # Load traces
    trace_dir = os.path.join(os.path.dirname(__file__), "..", "..", "work", "traces", "processed")

    # Use OLMoE traces (real inference)
    trace_file = os.path.join(trace_dir, "olmoe_traces_merged.npz")
    stats_file = os.path.join(trace_dir, "olmoe_trace_stats.json")

    if not os.path.exists(trace_file):
        print(f"ERROR: Trace file not found: {trace_file}")
        sys.exit(1)

    data = np.load(trace_file)
    traces = data["traces"]
    with open(stats_file) as f:
        stats = json.load(f)

    print("=" * 70)
    print("Trace Replay Harness — Workstream 1")
    print("=" * 70)
    print(f"Traces: {traces.shape[0]} tokens, {traces.shape[1]} layers, K={traces.shape[2]}")
    print(f"Source: OLMoE-1B-7B real inference")
    print()

    # Scale traces to Qwen3.5-397B dimensions
    # OLMoE: 16 layers, 64 experts, K=8
    # Qwen3.5-397B: 60 layers, 512 experts, K=3-10
    # We scale expert IDs by 512/64 = 8x and tile layers to 60
    print("Scaling OLMoE traces to Qwen3.5-397B dimensions...")
    scaled_traces = np.zeros((traces.shape[0], 60, 10), dtype=np.int32)
    for t in range(traces.shape[0]):
        for l in range(60):
            src_layer = l % traces.shape[1]  # tile 16 layers to 60
            for k in range(min(10, traces.shape[2])):
                # Scale expert ID: multiply by 8 and add random offset for diversity
                scaled_id = (int(traces[t, src_layer, k]) * 8 + (l // 16)) % 512
                scaled_traces[t, l, k] = scaled_id

    num_tokens = scaled_traces.shape[0]
    num_layers = 60
    num_experts = 512

    # Expert block size from Stage 0
    expert_size = int(3.5e6)  # 3.5 MB per expert at IQ2_XXS

    timing = TimingModel(expert_size)

    # Sweep configurations
    configs = [
        {"K": 3, "vram": 500, "ram": 3000, "label": "K=3, small cache"},
        {"K": 3, "vram": 1000, "ram": 5000, "label": "K=3, medium cache"},
        {"K": 3, "vram": 1500, "ram": 8000, "label": "K=3, large cache"},
        {"K": 5, "vram": 1000, "ram": 5000, "label": "K=5, medium cache"},
        {"K": 5, "vram": 1500, "ram": 8000, "label": "K=5, large cache"},
    ]

    results = []
    print(f"\n{'Label':<25} {'Causal':>8} {'Overlap':>8} {'SSD miss':>9} {'Cache hit':>9}")
    print("-" * 65)

    for cfg in configs:
        K = cfg["K"]
        cache = TieredCache(cfg["vram"], cfg["ram"], policy="lfu")

        # Build frequency map and init VRAM hotset
        freq = build_frequency_map(scaled_traces, K, num_layers)
        cache.init_vram_hotset(freq, num_layers, num_experts)

        # Replay
        r = replay_traces(scaled_traces, K, num_layers, num_experts, cache, timing)
        cs = r["cache_stats"]

        print(f"{cfg['label']:<25} {r['causal_tps']:>7.2f} {r['overlap_tps']:>7.2f} "
              f"{cs['ssd_miss_rate']:>8.1%} {cs['combined_cache_hit_rate']:>8.1%}")

        results.append({**cfg, **r, "cache_stats": cs})

    # Save
    outpath = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "replay_results.json")
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
