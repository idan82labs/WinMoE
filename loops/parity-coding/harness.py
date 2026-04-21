"""Parity harness — compare WinMoE logits vs llama.cpp logits position-by-position.

Inputs:
  --llama-bin     <path>    float32[n_pos * vocab] binary from llama-trace-dump --dump-logits
  --winmoe-bin    <path>    float32[n_pos * vocab] binary from WINMOE_DUMP_ALL_LOGITS
  --llama-json    <path>    sidecar JSON (vocab_size, all_ids)
  --winmoe-json   <path>    sidecar JSON
  --output-dir    <path>    results directory (plots + per-prompt JSON)
  --prompt-name   <str>     label for the plots
  --skip-prompt             skip prompt positions in metrics (compare only generated)
  --start-pos     <int>     first position to include in metrics (default: n_prompt, i.e. generated-only)
  --n-positions   <int>     limit (optional)

Outputs:
  output-dir/<prompt-name>_metrics.json    per-position arrays + summary
  output-dir/<prompt-name>_top1.png
  output-dir/<prompt-name>_top5.png
  output-dir/<prompt-name>_rank.png
  output-dir/<prompt-name>_delta.png

Metrics (per position):
  top1_agreement        argmax(winmoe) == argmax(llama)
  top5_overlap          |top5(winmoe) ∩ top5(llama)| / 5
  correct_token_rank    rank of argmax(llama) in winmoe sorted desc
  delta_winmoe          logit_winmoe[argmax(llama)] - logit_winmoe[argmax(winmoe)]
  delta_llama           same computed in llama.cpp (reference)
  kl_divergence         KL(softmax(winmoe) || softmax(llama))

Plots use rolling-10 smoothing where noted.
"""
import argparse, json, pathlib, sys
import numpy as np

def load(bin_path, json_path):
    meta = json.loads(pathlib.Path(json_path).read_text())
    vocab = meta["vocab_size"]
    n_pos = meta["n_positions"]
    data = np.fromfile(bin_path, dtype=np.float32)
    expected = n_pos * vocab
    if data.size != expected:
        print(f"WARN: {bin_path} has {data.size} floats, sidecar says {expected}", file=sys.stderr)
        n_pos = data.size // vocab
    arr = data[:n_pos * vocab].reshape(n_pos, vocab)
    return arr, meta

def softmax(logits, axis=-1):
    """Stable softmax."""
    m = logits.max(axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=axis, keepdims=True)

def kl_div(p, q, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float((p * (np.log(p) - np.log(q))).sum())

def compute_metrics(w, l):
    """w, l: [n_pos, vocab] logits arrays. Returns per-position dict of metric arrays."""
    n = min(w.shape[0], l.shape[0])
    w = w[:n]; l = l[:n]

    w_am = w.argmax(axis=1)
    l_am = l.argmax(axis=1)

    top1 = (w_am == l_am).astype(np.float32)

    # top5
    w_top5 = np.argpartition(w, -5, axis=1)[:, -5:]
    l_top5 = np.argpartition(l, -5, axis=1)[:, -5:]
    top5_overlap = np.zeros(n, dtype=np.float32)
    for i in range(n):
        top5_overlap[i] = len(set(w_top5[i].tolist()) & set(l_top5[i].tolist())) / 5.0

    # rank of llama's argmax in winmoe
    rank = np.zeros(n, dtype=np.int32)
    for i in range(n):
        target = int(l_am[i])
        wi = w[i]
        # rank = number of entries strictly greater
        rank[i] = int((wi > wi[target]).sum())

    # delta_winmoe = logit_winmoe[target] - logit_winmoe[winmoe_argmax]
    delta_w = np.zeros(n, dtype=np.float32)
    for i in range(n):
        delta_w[i] = w[i, int(l_am[i])] - w[i, int(w_am[i])]
    # delta_llama: we don't easily know llama's "correct" (would be teacher-forced self-argmax, which is 0)
    # Use: delta_llama at pos i = logit_llama[target] - logit_llama[llama_argmax] = 0 by def.
    # Instead, compute delta_llama = logit_llama at winmoe's argmax minus logit_llama at its own argmax.
    # That captures "how much llama disagrees with winmoe's pick".
    delta_l = np.zeros(n, dtype=np.float32)
    for i in range(n):
        delta_l[i] = l[i, int(w_am[i])] - l[i, int(l_am[i])]

    # KL(softmax(w) || softmax(l))
    kls = np.zeros(n, dtype=np.float32)
    for i in range(n):
        pw = softmax(w[i])
        pl = softmax(l[i])
        kls[i] = kl_div(pw, pl)

    return {
        "n_positions": n,
        "winmoe_argmax": w_am.tolist(),
        "llama_argmax":  l_am.tolist(),
        "top1_agreement": top1.tolist(),
        "top5_overlap":   top5_overlap.tolist(),
        "correct_token_rank": rank.tolist(),
        "delta_winmoe":   delta_w.tolist(),
        "delta_llama":    delta_l.tolist(),
        "kl_divergence":  kls.tolist(),
    }

def summary(m, start=0):
    """Reduce per-position arrays to scalar summary."""
    n = m["n_positions"]
    if start >= n:
        return {"error": f"start ({start}) >= n_positions ({n})"}
    s = slice(start, n)
    t1 = np.array(m["top1_agreement"][s])
    t5 = np.array(m["top5_overlap"][s])
    rk = np.array(m["correct_token_rank"][s])
    dw = np.array(m["delta_winmoe"][s])
    dl = np.array(m["delta_llama"][s])
    kl = np.array(m["kl_divergence"][s])
    n_scored = len(t1)
    return {
        "n_positions_scored": int(n_scored),
        "start_pos": start,
        "top1_agreement": float(t1.mean()),
        "top5_overlap":   float(t5.mean()),
        "rank_mean":      float(rk.mean()),
        "rank_p50":       int(np.percentile(rk, 50)),
        "rank_p95":       int(np.percentile(rk, 95)),
        "rank_le_1":      float((rk <= 1).mean()),
        "rank_le_3":      float((rk <= 3).mean()),
        "delta_winmoe_abs_mean": float(np.abs(dw).mean()),
        "delta_winmoe_abs_p95":  float(np.percentile(np.abs(dw), 95)),
        "delta_llama_abs_mean":  float(np.abs(dl).mean()),
        "delta_ratio_mean":      float((np.abs(dw) / (np.abs(dl) + 1e-6)).mean()),
        "kl_mean":    float(kl.mean()),
        "kl_p95":     float(np.percentile(kl, 95)),
    }

def plot_all(m, start, name, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots", file=sys.stderr)
        return []
    import numpy as np
    n = m["n_positions"]
    pos = np.arange(n)
    def rolling(a, w=10):
        a = np.array(a, dtype=np.float32)
        k = np.ones(w) / w
        return np.convolve(a, k, mode="valid")

    out = []
    # 1. top1
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(pos, m["top1_agreement"], lw=0.5, alpha=0.4, label="raw")
    if n >= 10:
        r = rolling(m["top1_agreement"])
        ax.plot(pos[5:5+len(r)], r, lw=1.5, label="rolling-10")
    ax.axvline(start, color='gray', lw=1, ls='--', alpha=0.6, label=f"start_pos={start}")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{name}: top-1 agreement vs llama.cpp")
    ax.set_xlabel("position")
    ax.legend()
    p = outdir / f"{name}_top1.png"
    fig.tight_layout(); fig.savefig(p); plt.close(fig); out.append(p)

    # 2. top5
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(pos, m["top5_overlap"], lw=0.5, alpha=0.4, label="raw")
    if n >= 10:
        r = rolling(m["top5_overlap"])
        ax.plot(pos[5:5+len(r)], r, lw=1.5, label="rolling-10")
    ax.axvline(start, color='gray', lw=1, ls='--', alpha=0.6)
    ax.axhline(0.7, color='red', lw=1, ls=':', alpha=0.5, label="gate 0.7")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{name}: top-5 overlap vs llama.cpp")
    ax.set_xlabel("position")
    ax.legend()
    p = outdir / f"{name}_top5.png"
    fig.tight_layout(); fig.savefig(p); plt.close(fig); out.append(p)

    # 3. rank
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(pos, m["correct_token_rank"], lw=0.8)
    ax.axvline(start, color='gray', lw=1, ls='--', alpha=0.6)
    ax.axhline(3, color='red', lw=1, ls=':', alpha=0.5, label="gate rank≤3")
    ax.set_yscale("symlog")
    ax.set_title(f"{name}: correct-token rank (lower is better)")
    ax.set_xlabel("position")
    ax.legend()
    p = outdir / f"{name}_rank.png"
    fig.tight_layout(); fig.savefig(p); plt.close(fig); out.append(p)

    # 4. delta overlay
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(pos, m["delta_winmoe"], lw=0.8, label="WinMoE delta", color='C0')
    ax.plot(pos, m["delta_llama"],  lw=0.8, label="llama.cpp delta", color='C1', alpha=0.7)
    ax.axvline(start, color='gray', lw=1, ls='--', alpha=0.6)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title(f"{name}: correct-token logit delta (logit[llama_argmax] - logit[self_argmax])")
    ax.set_xlabel("position")
    ax.legend()
    p = outdir / f"{name}_delta.png"
    fig.tight_layout(); fig.savefig(p); plt.close(fig); out.append(p)

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-bin",  required=True)
    ap.add_argument("--llama-json", required=True)
    ap.add_argument("--winmoe-bin", required=True)
    ap.add_argument("--winmoe-json",required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--prompt-name",required=True)
    ap.add_argument("--start-pos",  type=int, default=None,
                    help="first position to include in summary (default: n_prompt)")
    args = ap.parse_args()

    outdir = pathlib.Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    print(f"loading llama: {args.llama_bin}")
    l_arr, l_meta = load(args.llama_bin, args.llama_json)
    print(f"  shape={l_arr.shape} dtype={l_arr.dtype}")

    print(f"loading winmoe: {args.winmoe_bin}")
    w_arr, w_meta = load(args.winmoe_bin, args.winmoe_json)
    print(f"  shape={w_arr.shape} dtype={w_arr.dtype}")

    # Align lengths
    n = min(l_arr.shape[0], w_arr.shape[0])
    l_arr = l_arr[:n]; w_arr = w_arr[:n]

    # Defaults: score from n_prompt (i.e. generation-only)
    start = args.start_pos
    if start is None:
        start = int(l_meta.get("n_prompt", 0))

    print(f"computing metrics over {n} positions, summary starts at pos {start}")
    m = compute_metrics(w_arr, l_arr)
    s = summary(m, start=start)

    out = {
        "prompt_name": args.prompt_name,
        "llama_meta":  l_meta,
        "winmoe_meta": w_meta,
        "summary":     s,
        "per_position": m,
    }
    (outdir / f"{args.prompt_name}_metrics.json").write_text(json.dumps(out, indent=2))

    plots = plot_all(m, start, args.prompt_name, outdir)

    print("\n=== SUMMARY ===")
    for k, v in s.items():
        print(f"  {k}: {v}")

    # Gate check
    gate = True
    reasons = []
    if s.get("top1_agreement", 0) < 0.9:
        gate = False; reasons.append(f"top1_agreement={s['top1_agreement']:.3f} < 0.9")
    if s.get("top5_overlap", 0) < 0.7:
        gate = False; reasons.append(f"top5_overlap={s['top5_overlap']:.3f} < 0.7")
    if s.get("rank_le_3", 0) < 0.95:
        gate = False; reasons.append(f"rank_le_3={s['rank_le_3']:.3f} < 0.95")
    if s.get("delta_ratio_mean", 0) > 1.5:
        gate = False; reasons.append(f"delta_ratio_mean={s['delta_ratio_mean']:.3f} > 1.5")

    print(f"\nGATE: {'PASS' if gate else 'FAIL'}")
    for r in reasons:
        print(f"  FAIL: {r}")
    print(f"\nArtifacts: {len(plots)} plots + metrics.json in {outdir}")
    sys.exit(0 if gate else 1)

if __name__ == "__main__":
    main()
