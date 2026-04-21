"""Trial driver for speed-coding loop.

Usage:
  python run_trial.py <trial_id> <phase> "<change_description>" [--tokens N] [--code]

Runs WinMoE on the standard "Hello" chat benchmark, parses output, appends one row
to results.tsv. Optionally also runs the code benchmark.

Exit code 0 = trial completed and logged. 1 = build/run failed.
The KEEP/REJECT decision is left for human review (or computed via acceptance-rule.md).
"""
import os, re, sys, subprocess, time, datetime, pathlib

ROOT = pathlib.Path("C:/Users/idant/flash-moe-windows-v2/flash-moe-windows-v2-claude-filesystem-autoresearch-loops")
LOOP = ROOT / "loops" / "speed-coding"
TSV = LOOP / "results.tsv"
WINMOE_DIR = ROOT / "engine" / "runtime"
WINMOE_EXE = WINMOE_DIR / "winmoe.exe"
MODEL_397B = "D:/models/qwen35-397b-q4/Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-00001-of-00006.gguf"

CHAT_PROMPT_TOKENS = "248045,846,198,9419,248046,198,248045,74455,198"  # standard "Hello" chat


def kill_stale():
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command",
         "Get-Process winmoe -ErrorAction SilentlyContinue | Stop-Process -Force"],
        capture_output=True, timeout=15)
    time.sleep(2)


def build():
    r = subprocess.run(["cmd.exe", "/c", "C:/Users/idant/build_winmoe.bat"],
                       capture_output=True, text=True, timeout=300)
    return r.returncode == 0, r.stdout + r.stderr


def run_chat(tokens=30, env_extra=None):
    env = os.environ.copy()
    env["PATH"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin;" + env.get("PATH", "")
    env["OMP_NUM_THREADS"] = "2"
    env["MSYS_NO_PATHCONV"] = "1"
    if env_extra:
        env.update(env_extra)
    cmd = [str(WINMOE_EXE), "--model", MODEL_397B, "--tokens", str(tokens)]
    r = subprocess.run(cmd, cwd=str(WINMOE_DIR), env=env,
                       capture_output=True, text=True, timeout=900)
    return r.stdout + r.stderr


def parse(log):
    """Extract metrics from winmoe stdout/stderr."""
    m = {"tok_s_endtoend": "", "tok_s_steady": "", "attn_ms": "", "expert_ms": "",
         "lm_head_ms": "", "router_ms": "", "hit_rate": "", "think_logit": "",
         "coherent": "fail", "code_ok": "n/a"}
    # Tokens: 22 in 100614.2 ms = 0.22 tok/s
    g = re.search(r"Tokens:\s*(\d+)\s+in\s+([\d.]+)\s*ms\s*=\s*([\d.]+)\s*tok/s", log)
    if g:
        m["tok_s_endtoend"] = g.group(3)
    # Per-token times: collect tok 5..end
    pt = re.findall(r"Token\s+(\d+):\s+id=\d+\s+\(([\d.]+)\s*ms\)\s+\[attn=([\d.]+)\s+expert=([\d.]+)\s+router=([\d.]+)\]", log)
    if pt:
        gen = [t for t in pt if int(t[0]) >= 9]  # generated tokens (post-prompt)
        if gen:
            m["tok_s_steady"] = f"{1000.0 / (sum(float(t[1]) for t in gen) / len(gen)):.3f}"
            m["attn_ms"] = f"{sum(float(t[2]) for t in gen)/len(gen):.0f}"
            m["expert_ms"] = f"{sum(float(t[3]) for t in gen)/len(gen):.0f}"
            m["router_ms"] = f"{sum(float(t[4]) for t in gen)/len(gen):.0f}"
    lm = re.findall(r"LM_HEAD t\d+:\s*([\d.]+)\s*ms", log)
    if lm:
        m["lm_head_ms"] = f"{sum(float(x) for x in lm)/len(lm):.0f}"
    # Cache hit rate: 7466 hits, 5572 misses, 2824 stored (57.3% hit rate)
    g = re.search(r"\(([\d.]+)%\s+hit rate\)", log)
    if g:
        m["hit_rate"] = g.group(1)
    # think logit: Token 248068 (<think>) logit=20.9686
    g = re.search(r"Token 248068.*logit\s*=\s*(-?[\d.]+)", log)
    if g:
        m["think_logit"] = g.group(1)
    # Coherence: check generated tokens contain </think> and Hello in order.
    # (<think> is the predicted next-token at the LAST prompt position, marked [prompt]
    # in our logs, so we don't see it in gen_ids — but its presence is implied by
    # the next predicted being 271 = "\n\n" which only follows <think>.)
    gen_ids = [int(re.search(r"id=(\d+)", line).group(1))
               for line in log.split("\n")
               if "Token " in line and "id=" in line and "[prompt]" not in line]
    if 248069 in gen_ids and 9419 in gen_ids:
        i_close = gen_ids.index(248069)
        i_hello = gen_ids.index(9419)
        if i_hello > i_close:
            m["coherent"] = "pass"
    return m


def append(row):
    cols = ["timestamp", "trial", "phase", "change", "tok_s_endtoend", "tok_s_steady",
            "attn_ms", "expert_ms", "lm_head_ms", "router_ms", "hit_rate",
            "think_logit", "coherent", "code_ok", "decision", "notes"]
    line = "\t".join(str(row.get(c, "")) for c in cols)
    with open(TSV, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    if len(sys.argv) < 4:
        print("usage: run_trial.py <trial_id> <phase> '<change>' [--tokens N]", file=sys.stderr)
        sys.exit(2)
    trial_id, phase, change = sys.argv[1], sys.argv[2], sys.argv[3]
    tokens = 30
    for i, a in enumerate(sys.argv):
        if a == "--tokens" and i + 1 < len(sys.argv):
            tokens = int(sys.argv[i+1])

    print(f"[trial {trial_id}] phase={phase} change={change}")
    kill_stale()
    print("  building...")
    ok, build_log = build()
    if not ok:
        append({"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "trial": trial_id, "phase": phase, "change": change,
                "decision": "BUILD_FAIL", "notes": build_log[-200:].replace("\t", " ").replace("\n", " | ")})
        print("  BUILD FAILED")
        sys.exit(1)
    print("  running...")
    log = run_chat(tokens=tokens)
    m = parse(log)
    m["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    m["trial"] = trial_id
    m["phase"] = phase
    m["change"] = change
    m["decision"] = "PENDING"
    m["notes"] = ""
    append(m)
    print(f"  tok_s_endtoend={m['tok_s_endtoend']} steady={m['tok_s_steady']} "
          f"attn={m['attn_ms']} expert={m['expert_ms']} lm_head={m['lm_head_ms']} "
          f"think={m['think_logit']} coherent={m['coherent']}")
    sys.exit(0)


if __name__ == "__main__":
    main()
