"""
WinMoE Engine Benchmark — IMMUTABLE
DO NOT MODIFY THIS FILE.

Runs the custom engine, measures tok/s, outputs JSON for autoresearch.
"""
import subprocess
import time
import json
import sys
import os
import re

ENGINE = os.path.join(os.path.dirname(__file__), "engine.exe")
PROMPT = "Explain the theory of general relativity and its implications for modern physics."
NUM_TOKENS = 50
BASELINE_TPS = 1.8  # llama.cpp ceiling to beat

def run():
    if not os.path.exists(ENGINE):
        print(json.dumps({"tok_s": 0, "status": "no_binary"}))
        return

    cmd = [ENGINE, "--prompt", PROMPT, "--tokens", str(NUM_TOKENS)]

    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=300, encoding='utf-8', errors='replace')
    except subprocess.TimeoutExpired:
        print(json.dumps({"tok_s": 0, "status": "timeout"}))
        return
    except Exception as e:
        print(json.dumps({"tok_s": 0, "status": f"error: {e}"}))
        return

    wall = time.perf_counter() - t0

    # Parse engine output for timing
    output = result.stdout + "\n" + result.stderr

    tok_s = 0
    first_token_ms = 0
    tokens_generated = 0

    # Look for JSON timing line
    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("{") and "tok_s" in line:
            try:
                data = json.loads(line)
                tok_s = data.get("tok_s", 0)
                first_token_ms = data.get("first_token_ms", 0)
                tokens_generated = data.get("tokens", 0)
                break
            except json.JSONDecodeError:
                pass

    # Fallback: calculate from wall time
    if tok_s == 0 and wall > 0:
        # Count generated tokens from output
        generated = result.stdout.strip()
        if generated:
            # Rough: each word ≈ 1.3 tokens
            tokens_generated = len(generated.split())
            if tokens_generated > 0 and wall > 1:
                tok_s = tokens_generated / wall

    improvement = ((tok_s - BASELINE_TPS) / BASELINE_TPS * 100) if BASELINE_TPS > 0 else 0

    res = {
        "tok_s": round(tok_s, 2),
        "first_token_ms": round(first_token_ms, 1),
        "tokens": tokens_generated,
        "wall_s": round(wall, 1),
        "vs_baseline": f"{improvement:+.1f}%",
        "status": "ok" if tok_s > 0 else "no_output",
        "output_sample": result.stdout[:200] if result.stdout else "",
    }

    print(json.dumps(res, indent=2))

    # Summary to stderr
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"  tok/s: {tok_s:.2f} (vs baseline {BASELINE_TPS}: {improvement:+.1f}%)", file=sys.stderr)
    print(f"  wall: {wall:.1f}s, tokens: {tokens_generated}", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)

if __name__ == "__main__":
    run()
