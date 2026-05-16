"""Phase 3: free-gen code-quality validation.

For each of the 5 prompts:
  1. Run WinMoE with --prompt-tokens=<prompt> --tokens=<n_prompt+100>
  2. Parse stdout for the 100 generated IDs (after prompt)
  3. Decode IDs -> text via GGUF tokenizer
  4. Strip chat-template wrappers (<think>...</think> etc.)
  5. Apply parser: ast.parse for code prompts, json.loads for JSON prompts
  6. Pass/fail per prompt; aggregate
"""
import os, sys, json, pathlib, subprocess, re, ast
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path("C:/Users/idant/flash-moe-windows-v2/flash-moe-windows-v2-claude-filesystem-autoresearch-loops")
LOOP = ROOT / "loops" / "parity-coding"
PREFIX_DIR = LOOP / "prefixes"
RESULTS_DIR = LOOP / "results" / "phase3"
WINMOE_DIR = ROOT / "engine" / "runtime"
WINMOE_EXE = WINMOE_DIR / "winmoe.exe"
MODEL = "D:/models/qwen35-397b-q4/Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-00001-of-00006.gguf"
TMP = pathlib.Path("C:/Users/idant")
VOCAB_PATH = TMP / "qwen35_vocab.json"

# Per-prompt parser test
PARSERS = {
    1: "python",   # bracket_closure — Python dict literal assignment
    2: "python",   # indentation_python
    3: "python",   # var_disambiguation
    4: "python",   # operator_choice
    5: "json",     # json_output — pure JSON
}

def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        v = json.load(f)
    return {int(k): val for k, val in v.items()}

def decode(ids, vocab):
    s = "".join(vocab.get(i, "?") for i in ids)
    # Qwen3.5 BPE uses Ġ for space, Ċ for newline (standard GPT-style)
    return s.replace("Ġ", " ").replace("Ċ", "\n")

def close_brackets(s):
    """Append closing brackets so a truncated snippet can be parsed."""
    stack = []
    pair = {"(": ")", "[": "]", "{": "}"}
    in_str = None
    for c in s:
        if in_str:
            if c == in_str: in_str = None
            continue
        if c in ('"', "'"): in_str = c; continue
        if c in pair: stack.append(pair[c])
        elif c in ")]}" and stack and stack[-1] == c: stack.pop()
    if in_str: s += in_str
    return s + "".join(reversed(stack))

def run_winmoe(prompt_id, name, prompt_ids_str, n_prompt, n_gen=100):
    out_log = TMP / f"phase3_{prompt_id:02d}_winmoe.log"
    if out_log.exists() and out_log.stat().st_size > 1000:
        return out_log
    env = os.environ.copy()
    env["PATH"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin;" + env.get("PATH","")
    env["OMP_NUM_THREADS"] = "2"
    env["WINMOE_TOPK"] = "10"
    cmd = [
        str(WINMOE_EXE),
        "--model", MODEL,
        "--prompt-tokens", prompt_ids_str,
        "--tokens", str(n_prompt + n_gen),
    ]
    print(f"  running winmoe (prompt {n_prompt} + gen {n_gen})...")
    with open(out_log, "w") as lf:
        r = subprocess.run(cmd, cwd=str(WINMOE_DIR), env=env,
                           stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    print(f"  rc={r.returncode}")
    return out_log if r.returncode == 0 else None

def extract_gen_ids(log_path, n_prompt):
    """WinMoE prints each token id followed by a space to stdout via printf in the main loop.
    Total emitted = num_tokens. First n_prompt are prompt; remainder are generated."""
    txt = pathlib.Path(log_path).read_text(errors="replace")
    # Token lines look like: "<id> " — but they're interleaved with stderr lines because both go to log_path.
    # Find lines that are JUST a stream of integers separated by spaces (the stdout printf stream).
    # In practice it's the "Token N: id=X (...) [...]" lines that are the trustworthy source.
    ids = []
    for m in re.finditer(r"^Token (\d+): id=(\d+)", txt, re.MULTILINE):
        ids.append(int(m.group(2)))
    return ids

def strip_chat_template(text):
    """Strip <think>...</think> blocks and surrounding wrappers."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|im_(start|end)\|>.*?(?=\n|$)", "", text)
    return text.strip()

def try_python(code):
    """Try parsing raw, then with closed brackets, then progressively trim trailing lines."""
    last_err = None
    # First try the raw code, then increasingly truncated versions
    lines = code.rstrip().split("\n")
    for k in range(0, min(len(lines), 5)):
        trimmed = "\n".join(lines[:len(lines)-k]) if k else code
        for attempt in (trimmed, close_brackets(trimmed)):
            try:
                ast.parse(attempt)
                return True, None
            except SyntaxError as e:
                last_err = f"line {e.lineno}: {e.msg}"
    return False, last_err

def try_json(text):
    """Try to parse JSON: prefer fenced ```json blocks, fall back to first '{' or '['."""
    candidates = []
    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)(?:```|$)", text, re.DOTALL):
        candidates.append(m.group(1).strip())
    # also try slicing from each { or [ in the text
    for i, c in enumerate(text):
        if c in "{[":
            candidates.append(text[i:])
    last_err = None
    for sub in candidates:
        for attempt in (sub, close_brackets(sub)):
            try:
                return True, json.loads(attempt)
            except json.JSONDecodeError as e:
                last_err = str(e)
    return False, last_err or "no valid JSON object found"

def extract_code_block(text, lang_hint=None):
    """Pull out ```python ... ``` or ``` ... ``` fenced block, or {...} for JSON."""
    m = re.search(r"```(?:python|json)?\s*\n(.*?)```", text, re.DOTALL)
    if m: return m.group(1)
    return text

def main():
    pdef = json.loads((LOOP / "prompts.json").read_text())
    vocab = load_vocab()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "phase3_results.json"

    rows = []
    for p in pdef["prompts"]:
        pid = p["id"]
        name = p["name"]
        print(f"\n=== P{pid} {name} ===")
        ids_path = PREFIX_DIR / f"{pid:02d}_{name}.tokens"
        ids = ids_path.read_text().strip()
        n_prompt = len(ids.split(","))
        print(f"  n_prompt={n_prompt}")

        log = run_winmoe(pid, name, ids, n_prompt, n_gen=100)
        if not log: continue
        all_ids = extract_gen_ids(log, n_prompt)
        gen_ids = all_ids[n_prompt:]
        text = decode(gen_ids, vocab)
        body = strip_chat_template(text)
        parser = PARSERS[pid]
        if parser == "python":
            block = extract_code_block(body, "python")
            ok, err = try_python(block)
        else:
            ok, err = try_json(body)
        print(f"  parser={parser} ok={ok}  err={err}")
        print(f"  decoded text (first 400 chars):\n    {text[:400]!r}")
        rows.append({
            "prompt_id": pid, "name": name, "parser": parser,
            "n_gen": len(gen_ids), "ok": bool(ok), "err": err,
            "text": text[:1000]
        })

    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_json}")

    syntax_count = sum(1 for r in rows if r["ok"])
    json_rows = [r for r in rows if r["parser"] == "json"]
    json_pass = sum(1 for r in json_rows if r["ok"])
    print(f"\n=== Phase 3 SUMMARY ===")
    print(f"  syntax_valid: {syntax_count}/5")
    print(f"  json_valid:   {json_pass}/{len(json_rows)}")
    gate = (syntax_count >= 4) and (json_pass == len(json_rows))
    print(f"  GATE: {'PASS' if gate else 'FAIL'}")
    sys.exit(0 if gate else 1)

if __name__ == "__main__":
    main()
