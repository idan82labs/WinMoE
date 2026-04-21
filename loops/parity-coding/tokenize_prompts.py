"""Tokenize the 5 parity prompts via llama-trace-dump --tokenize.

Reads prompts.json, wraps each body in the chat format, calls the tokenizer,
writes per-prompt token id files into prefixes/<id>_<name>.tokens
"""
import json, os, subprocess, pathlib, sys

ROOT = pathlib.Path("C:/Users/idant/flash-moe-windows-v2/flash-moe-windows-v2-claude-filesystem-autoresearch-loops")
LOOP = ROOT / "loops" / "parity-coding"
PREFIX_DIR = LOOP / "prefixes"
PROMPTS = LOOP / "prompts.json"
TRACE_DUMP = pathlib.Path("D:/llama-cpp-src/build-cuda/bin/llama-trace-dump.exe")
# Use 35B for tokenization - same Qwen3.5 vocab, smaller load
MODEL_FOR_TOKENIZER = "D:/models/qwen35-35b-q4/Qwen3.5-35B-A3B-Q4_K_M.gguf"

def main():
    cfg = json.loads(PROMPTS.read_text())
    wrapper = cfg["chat_wrapper"]
    env = os.environ.copy()
    env["PATH"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin;" + env.get("PATH", "")

    for p in cfg["prompts"]:
        full_text = wrapper.replace("{body}", p["body"])
        out_file = PREFIX_DIR / f"{p['id']:02d}_{p['name']}.tokens"
        tmp = PREFIX_DIR / f"_tok_tmp_{p['id']}.txt"

        cmd = [
            str(TRACE_DUMP),
            "-m", MODEL_FOR_TOKENIZER,
            "-ngl", "0",
            "-o", str(tmp),
            "--tokenize", full_text,
        ]
        print(f"[{p['id']}] {p['name']}: tokenizing...")
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            print(f"  FAILED rc={r.returncode}: {r.stderr[-500:]}", file=sys.stderr)
            continue

        # Tmp file format:
        # # tokenize result for: <text-with-possibly-embedded-newlines>
        # 248045,846,...
        # Find the LAST line that looks like comma-separated digits.
        text = tmp.read_text()
        import re
        candidates = [ln.strip() for ln in text.split("\n")
                      if re.fullmatch(r"-?\d+(?:,-?\d+)*", ln.strip())]
        if not candidates:
            print(f"  no ids parsed: {text[:200]!r}", file=sys.stderr)
            continue
        ids_line = candidates[-1]
        out_file.write_text(ids_line)
        print(f"  -> {out_file.name}: {len(ids_line.split(','))} tokens")
        tmp.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
