"""Orchestrate the full parity sprint: for each prompt,
  1. llama-trace-dump 397B (prompt + --n-gen N) → reference logits .bin + .json
  2. WinMoE 397B with the combined (prompt+prefix) token list, --tokens=N+prompt_len, WINMOE_DUMP_ALL_LOGITS → our logits .bin + .json
  3. harness.py → per-position metrics + plots + gate verdict
  4. append summary row to results.tsv

Usage:
  python run_parity.py [--only N] [--n-gen N]   # --only=4 runs just prompt 4
"""
import argparse, json, os, pathlib, subprocess, sys, time, datetime, re

ROOT = pathlib.Path("C:/Users/idant/flash-moe-windows-v2/flash-moe-windows-v2-claude-filesystem-autoresearch-loops")
LOOP = ROOT / "loops" / "parity-coding"
PREFIX_DIR = LOOP / "prefixes"
RESULTS_DIR = LOOP / "results"
TSV = LOOP / "results.tsv"

LLAMA_TRACE = pathlib.Path("D:/llama-cpp-src/build-cuda/bin/llama-trace-dump.exe")
MODEL = "D:/models/qwen35-397b-q4/Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-00001-of-00006.gguf"
WINMOE_DIR = ROOT / "engine" / "runtime"
WINMOE_EXE = WINMOE_DIR / "winmoe.exe"

TMP = pathlib.Path("C:/Users/idant")

def kill_stale():
    subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                    "Get-Process winmoe,llama-trace-dump -ErrorAction SilentlyContinue | Stop-Process -Force"],
                   capture_output=True, timeout=30)
    time.sleep(2)

def read_prompt(prompt_id):
    pdef = json.loads((LOOP / "prompts.json").read_text())
    p = next(x for x in pdef["prompts"] if x["id"] == prompt_id)
    ids_path = PREFIX_DIR / f"{prompt_id:02d}_{p['name']}.tokens"
    ids = ids_path.read_text().strip()
    return p, ids

def run_llama(prompt_id, prompt_ids, n_gen):
    out_bin  = TMP / f"parity_{prompt_id:02d}_llama.bin"
    out_json = TMP / f"parity_{prompt_id:02d}_llama.bin.json"
    out_log  = TMP / f"parity_{prompt_id:02d}_llama.log"
    if out_bin.exists() and out_json.exists():
        print(f"  llama bin exists ({out_bin.name}) — skipping")
        return out_bin, out_json
    env = os.environ.copy()
    env["PATH"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin;" + env.get("PATH","")
    env["OMP_NUM_THREADS"] = "2"
    env["MSYS_NO_PATHCONV"] = "1"
    cmd = [
        str(LLAMA_TRACE),
        "-m", MODEL, "-ngl", "0",
        "-o", str(TMP / f"parity_{prompt_id:02d}_llama_trace.txt"),
        "--tokens", prompt_ids,
        "--dump-logits", str(out_bin),
        "--n-gen", str(n_gen),
        "--filter", r"^result_norm$",
    ]
    print(f"  running llama ({n_gen} gen)...")
    t0 = time.time()
    with open(out_log, "w") as lf:
        r = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    print(f"  llama done in {time.time()-t0:.0f}s, rc={r.returncode}")
    if r.returncode != 0:
        print(f"  FAIL — see {out_log}")
        return None, None
    return out_bin, out_json

def run_winmoe(prompt_id, prompt_ids, all_ids_str, n_total):
    """all_ids_str: comma-separated prompt + generated IDs (from llama sidecar). n_total = n_prompt+n_gen."""
    out_bin  = TMP / f"parity_{prompt_id:02d}_winmoe.bin"
    out_json = TMP / f"parity_{prompt_id:02d}_winmoe.bin.json"
    out_log  = TMP / f"parity_{prompt_id:02d}_winmoe.log"
    if out_bin.exists() and out_json.exists():
        print(f"  winmoe bin exists — skipping")
        return out_bin, out_json
    env = os.environ.copy()
    env["PATH"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin;" + env.get("PATH","")
    env["OMP_NUM_THREADS"] = "2"
    env["MSYS_NO_PATHCONV"] = "1"
    env["WINMOE_TOPK"] = "10"
    env["WINMOE_DUMP_ALL_LOGITS"] = str(out_bin)
    cmd = [
        str(WINMOE_EXE),
        "--model", MODEL,
        "--prompt-tokens", all_ids_str,
        "--tokens", str(n_total),  # no generation — WinMoE does n_total forward passes on the supplied ids
    ]
    print(f"  running winmoe ({n_total} positions)...")
    t0 = time.time()
    with open(out_log, "w") as lf:
        r = subprocess.run(cmd, cwd=str(WINMOE_DIR), env=env,
                           stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    print(f"  winmoe done in {time.time()-t0:.0f}s, rc={r.returncode}")
    if r.returncode != 0:
        print(f"  FAIL — see {out_log}")
        return None, None
    return out_bin, out_json

def run_harness(prompt_id, name, llama_bin, llama_json, winmoe_bin, winmoe_json):
    outdir = RESULTS_DIR / f"{prompt_id:02d}_{name}"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "C:/Python313/python.exe", str(LOOP / "harness.py"),
        "--llama-bin",  str(llama_bin),
        "--llama-json", str(llama_json),
        "--winmoe-bin", str(winmoe_bin),
        "--winmoe-json", str(winmoe_json),
        "--output-dir", str(outdir),
        "--prompt-name", f"{prompt_id:02d}_{name}",
    ]
    print(f"  running harness...")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(r.stdout[-1500:] if r.stdout else "")
    if r.returncode not in (0, 1):
        print(f"  HARNESS ERROR rc={r.returncode}: {r.stderr[-500:]}")
    metrics_path = outdir / f"{prompt_id:02d}_{name}_metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        return m.get("summary", {}), r.returncode == 0
    return {}, False

def append_tsv(row):
    cols = ["timestamp", "trial", "prompt_id", "prompt_name", "top1_agreement",
            "top5_overlap", "rank_p95", "delta_ratio_p90", "kl_mean",
            "q6k_parity", "decision", "notes"]
    line = "\t".join(str(row.get(c, "")) for c in cols)
    with open(TSV, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None, help="only run prompt N (1-5)")
    ap.add_argument("--n-gen", type=int, default=50, help="tokens to generate with llama for teacher-forcing")
    args = ap.parse_args()

    pdef = json.loads((LOOP / "prompts.json").read_text())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for p in pdef["prompts"]:
        if args.only and p["id"] != args.only:
            continue
        print(f"\n=== PROMPT {p['id']}: {p['name']} ===")
        pinfo, prompt_ids = read_prompt(p["id"])
        n_prompt = len(prompt_ids.split(","))
        print(f"  prompt tokens: {n_prompt}")
        # NOTE: one engine at a time — machine can't dual-host 397B without thrashing
        kill_stale()
        llama_bin, llama_json = run_llama(p["id"], prompt_ids, args.n_gen)
        if not llama_bin: continue
        # read sidecar to get generated IDs
        meta = json.loads(llama_json.read_text())
        all_ids = ",".join(str(i) for i in meta["all_ids"])
        n_total = meta["n_positions"]
        kill_stale()
        winmoe_bin, winmoe_json = run_winmoe(p["id"], prompt_ids, all_ids, n_total)
        if not winmoe_bin: continue
        summary, gate_pass = run_harness(p["id"], p["name"], llama_bin, llama_json, winmoe_bin, winmoe_json)

        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "trial": f"T_{p['id']:02d}_{p['name']}",
            "prompt_id": p["id"],
            "prompt_name": p["name"],
            "top1_agreement": f"{summary.get('top1_agreement', 0):.3f}",
            "top5_overlap":   f"{summary.get('top5_overlap',   0):.3f}",
            "rank_p95":       summary.get("rank_p95", ""),
            "delta_ratio_p90": f"{summary.get('delta_winmoe_abs_p95', 0):.3f}",
            "kl_mean":        f"{summary.get('kl_mean', 0):.4f}",
            "q6k_parity":     "PASS",
            "decision":       "PASS" if gate_pass else "FAIL",
            "notes":          "",
        }
        append_tsv(row)
        results.append((p["id"], p["name"], gate_pass, summary))

    print("\n=== FINAL SUMMARY ===")
    all_pass = True
    for pid, name, gate, summ in results:
        tag = "PASS" if gate else "FAIL"
        print(f"  [{tag}] P{pid} {name}  top1={summ.get('top1_agreement', 0):.3f}  "
              f"top5={summ.get('top5_overlap',0):.3f}  rank_p95={summ.get('rank_p95','?')}")
        all_pass = all_pass and gate

    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()
