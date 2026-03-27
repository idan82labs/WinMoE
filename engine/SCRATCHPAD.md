# Engine Build Scratchpad — Read This After Compaction

## WHAT TO DO NEXT
Build the Expert I/O Reader (component 2 of 5). Slab repacker is DONE.

## COMPLETED
- [x] Slab repacker: `engine/repacker/repack_experts.py` — 114.6 GB slab at D:/flash-moe-engine/experts.slab
  - 30,720 slots (60 layers × 512 experts), 3,735,552 bytes/slot (57 × 64 KiB)
  - Index at D:/flash-moe-engine/expert_index.json
  - Benchmark: 1820 MB/s explicit unbuffered reads on slab

## LOCKED SLAB FORMAT
- Single file, NOT per-layer files
- 64 KiB slot alignment
- Layer-major order (all experts for layer 0 first, then layer 1, etc.)
- Within each layer: experts sorted by hotness (most frequent first)
- Hotness comes from OLMoE trace stats or static frequency analysis
- In-RAM index: JSON file mapping (layer, expert_id) → file offset

## REPACKER INPUT
GGUF shards at:
```
D:/hf_cache/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/
  da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_XXS/
  Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf  (tiny, header)
  Qwen3.5-397B-A17B-UD-IQ2_XXS-00002-of-00004.gguf  (49.8 GB)
  Qwen3.5-397B-A17B-UD-IQ2_XXS-00003-of-00004.gguf  (49.9 GB)
  Qwen3.5-397B-A17B-UD-IQ2_XXS-00004-of-00004.gguf  (15.1 GB)
```

## EXPERT TENSOR NAMES IN GGUF
Expert weights are stored as stacked tensors per layer:
- `blk.{L}.ffn_gate_exps.weight` — shape [n_embd, n_ff, n_expert] = [4096, 1024, 512]
- `blk.{L}.ffn_up_exps.weight` — same shape
- `blk.{L}.ffn_down_exps.weight` — shape [n_ff, n_embd, n_expert] = [1024, 4096, 512]

OR merged:
- `blk.{L}.ffn_gate_up_exps.weight` — shape [n_embd, n_ff*2, n_expert]

Each expert's data = slice along the n_expert dimension.

## REPACKER OUTPUT
- `D:/flash-moe-engine/experts.slab` — single slab file
- `D:/flash-moe-engine/expert_index.json` — offset index
- `D:/flash-moe-engine/shared_weights.bin` — non-expert weights (attention, embed, norms, routing)

## BUILD ORDER AFTER REPACKER
2. Expert I/O reader (C++) — engine/io/
3. Tiered cache manager (C++) — engine/cache/
4. Expert streaming scheduler (C++) — engine/scheduler/
5. Integration — engine/runtime/

## KEY CONSTRAINT
D: drive has ~170 GB free. Slab file will be ~107 GB (same as GGUF total).
After repacking, can delete GGUF cache to reclaim 107 GB.
Total disk needed during repacking: 107 (GGUF) + 107 (slab) = 214 GB. Tight but fits.
