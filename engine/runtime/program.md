# WinMoE Custom Engine — Autoresearch Program

## Goal
Build a standalone MoE inference engine that achieves 3+ tok/s (target 5) on
Qwen3.5-397B-A17B IQ2_XXS by replacing llama.cpp's mmap with explicit I/O,
tiered caching, and within-layer expert streaming.

## Metric
Primary: `tok/s` (generation tokens per second, measured over 50+ tokens)
Secondary: `first_token_ms` (time to first token)
Guardrail: output must be coherent English (spot-check)

## What You Can Modify
- `engine.cpp` — the inference engine (model loading, forward pass, I/O)
- `engine.h` — engine headers
- Any file in `engine/runtime/`

## What You Cannot Modify
- `benchmark.py` — the evaluation harness (immutable)
- Slab file format (locked by professor)
- Expert index format (locked)
- The GGUF model files

## Architecture (Professor-Locked)
1. Shared weights (attention, embeddings, norms, routing): loaded into RAM at startup
2. Expert weights: read from `D:/flash-moe-engine/experts.slab` via explicit unbuffered I/O
3. Tiered cache: RAM primary (8,571 blocks), VRAM staging (150-300 blocks)
4. Within-layer streaming: run gate → sort experts by residency → compute first ready → stream rest
5. No mmap. No OS page cache dependency. Hard user-space memory budget.

## Key Numbers
- Expert block: 3,735,552 bytes (3.56 MB), 64 KiB aligned
- Per-expert compute: ~1.71 ms
- Per-expert SSD read: ~1.67 ms (at 2,031 MB/s measured)
- Per-expert H2D: ~0.18 ms (at 19 GB/s pinned)
- 60 layers, 512 experts/layer, K=3 active
- Shared weights: ~7 GB
- Model: IQ2_XXS quantization (2-bit importance-matrix)

## Experiment Loop
Each experiment:
1. Modify engine code (one change)
2. Build: `cl /O2 /EHsc /std:c++17 engine.cpp /Fe:engine.exe`
3. Run: `python benchmark.py` (measures tok/s over fixed prompt)
4. If tok/s improved by >2%: keep (git commit)
5. If regressed or crashed: revert (git checkout)
6. Log to `results.tsv`
7. Do NOT pause to ask — keep iterating

## Build Order (Phase 1: Get Any Output)
1. Parse expert_index.json to get slab offsets
2. Load shared weights from GGUF (attention, embeddings, norms, routing)
3. Implement tokenizer (or shell out to llama-cli for tokenization)
4. Implement forward pass for ONE layer:
   a. Attention (simplified — can use dummy initially)
   b. RMSNorm
   c. Router (linear projection → softmax → topK)
   d. Expert FFN: read from slab → dequant → gate_proj × up_proj → SwiGLU → down_proj
   e. Expert combine + residual
5. Loop over 60 layers
6. Final norm → LM head → sample token
7. Measure tok/s

## Build Order (Phase 2: Optimize)
- Replace dummy attention with real GQA attention
- Add KV cache
- Add tiered expert cache (RAM LFU)
- Add within-layer streaming (async I/O + compute overlap)
- Optimize dequant kernels (CUDA or AVX512 intrinsics)
- Profile and eliminate bottlenecks

## Simplicity Rules
- Start with the simplest possible implementation that generates tokens
- Use CPU-only first, add GPU later
- Use existing libraries where possible (ggml for tensor ops)
- Prefer 10 lines of working code over 100 lines of elegant code
- Each experiment should be one focused change

## What NOT to Do
- Don't build a chat interface (just generate tokens)
- Don't implement speculative decoding
- Don't implement batch inference
- Don't optimize attention before expert I/O is proven faster than mmap
- Don't use mmap for anything
