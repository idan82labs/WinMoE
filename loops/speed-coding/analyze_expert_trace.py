"""Analyze WinMoE expert-access trace to decide hot/cold expert split for
IQ2 cold-tier quantization (Phase 4.1).

Input: TSV from WINMOE_EXPERT_TRACE — layer<TAB>expert<TAB>hits.
Output:
  - per-layer histogram (hits sorted desc)
  - cumulative-hits curve
  - recommended cold-set (bottom X% experts that account for Y% of misses)
  - JSON file with the hot/cold split per layer
"""
import sys, json, pathlib, collections
from typing import Dict, List, Tuple
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def parse(path):
    data = collections.defaultdict(dict)  # layer -> {expert: hits}
    for ln in pathlib.Path(path).read_text().splitlines():
        if not ln or ln.startswith("#"): continue
        l, e, h = ln.split("\t")
        data[int(l)][int(e)] = int(h)
    return data

def analyze(data, cold_pct=0.5):
    """Pick the bottom `cold_pct` of experts per layer by access count.
    A 50% cold split with IQ2_XS (~2.5 bpw vs Q4_K_M ~4.5 bpw) cuts per-miss
    bytes ~1.8×; combined with 50% of misses hitting cold tier → ~30% bytes saved."""
    summary = {}
    for layer, hits in sorted(data.items()):
        n_experts_seen = len(hits)
        # Pad zeroes for un-accessed experts (assume they're cold too)
        total_experts = 512  # for 397B; configurable
        all_hits = [(e, hits.get(e, 0)) for e in range(total_experts)]
        # sort ascending by hits — coldest first
        all_hits.sort(key=lambda x: x[1])
        n_cold = int(total_experts * cold_pct)
        cold_set = sorted(e for e, h in all_hits[:n_cold])
        hot_set = sorted(e for e, h in all_hits[n_cold:])
        total_hits = sum(hits.values())
        cold_hits = sum(hits.get(e, 0) for e in cold_set)
        cold_hit_share = cold_hits / total_hits if total_hits else 0
        summary[layer] = {
            "n_experts_accessed": n_experts_seen,
            "total_hits": total_hits,
            "cold_set": cold_set,
            "hot_set": hot_set,
            "cold_hit_share": cold_hit_share,
        }
    return summary

def main():
    if len(sys.argv) < 2:
        print("usage: analyze_expert_trace.py <expert_trace.tsv> [cold_pct=0.5] [out_json=expert_split.json]")
        sys.exit(1)
    path = sys.argv[1]
    cold_pct = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    out_path = sys.argv[3] if len(sys.argv) > 3 else "expert_split.json"

    data = parse(path)
    print(f"Loaded {sum(len(d) for d in data.values())} (layer,expert) entries across {len(data)} layers")
    summary = analyze(data, cold_pct=cold_pct)

    # Per-layer report
    print(f"\n=== Per-layer ({cold_pct*100:.0f}% cold) ===")
    avg_cold_share = 0
    for layer, s in sorted(summary.items()):
        print(f"  L{layer:2d}: {s['n_experts_accessed']:3d}/{512} experts seen, "
              f"{s['total_hits']:5d} hits, cold-set absorbs {s['cold_hit_share']*100:5.1f}% of misses")
        avg_cold_share += s["cold_hit_share"]
    avg_cold_share /= max(len(summary), 1)
    print(f"\n  avg cold-set hit share: {avg_cold_share*100:.2f}%")
    print(f"  hot-set hit share:      {(1 - avg_cold_share)*100:.2f}%")
    print(f"  expected SSD-IO reduction if cold→IQ2 (2.5/4.5 = 0.56× size):")
    bytes_saved = avg_cold_share * (1 - 0.56)
    print(f"    ~{bytes_saved*100:.1f}% bytes saved on cold-tier reads alone")

    # Write JSON
    out = {
        "cold_pct": cold_pct,
        "total_experts": 512,
        "avg_cold_hit_share": avg_cold_share,
        "layers": summary,
    }
    pathlib.Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\nWrote split to {out_path}")

if __name__ == "__main__":
    main()
