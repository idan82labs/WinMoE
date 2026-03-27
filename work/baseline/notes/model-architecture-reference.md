# Model Architecture Reference — Qwen3 / Qwen3.5 MoE

## Primary target: Qwen3.5-397B-A17B (matches spec K=10, ~6.75 MB/expert at Q4)

| Parameter | Qwen3-235B-A22B | Qwen3.5-397B-A17B |
|-----------|------------------|--------------------|
| Total params | ~235B | ~397B |
| Active params | ~22B | ~17B |
| Layers | 94 (all MoE) | 60 (all MoE) |
| Hidden size | 4096 | 4096 |
| Experts per layer | 128 routed, 0 shared | 512 routed, 1 shared |
| K (active routed) | 8 | 10 |
| Total active/token | 8 | 10 + 1 shared = 11 |
| Expert intermediate | 1536 | 1024 |
| Expert params | 18.87M | 12.58M |

## Expert block sizes

### Qwen3.5-397B-A17B (primary target)
| Quantization | Per expert | K=10 per-layer miss | K=11 per-layer (incl shared) |
|-------------|-----------|---------------------|-------------------------------|
| FP16 | 25.17 MB | 251.7 MB | 276.9 MB |
| Q8 | 12.58 MB | 125.8 MB | 138.4 MB |
| Q4 | **6.29 MB** | **62.9 MB** | **69.2 MB** |

The spec's "6.75 MB per expert, K=10, D≈67.5 MB" aligns with **Qwen3.5-397B at Q4** quantization.

### Qwen3-235B-A22B (secondary target)
| Quantization | Per expert | K=8 per-layer miss |
|-------------|-----------|---------------------|
| FP16 | 37.75 MB | 302.0 MB |
| Q8 | 18.87 MB | 150.96 MB |
| Q4 | 9.44 MB | 75.5 MB |

## Total model sizes (expert weights only)

| Model | Experts total | Q4 total expert bytes |
|-------|--------------|----------------------|
| Qwen3.5-397B | 512 × 60 = 30,720 experts | 30,720 × 6.29 MB = **188.7 GB** |
| Qwen3-235B | 128 × 94 = 12,032 experts | 12,032 × 9.44 MB = **110.8 GB** |

## Routing statistics (from Qwen3-30B analysis)
- Expert utilization: 14.3× ratio between most and least used
- Gini coefficient: 0.30 (moderate imbalance)
- Top-10 experts handle ~17.7% of tokens
- Task-specific concentration observed (math experts, code experts)

## Sources
- Qwen3-235B-A22B config.json (HuggingFace)
- Qwen3.5-397B-A17B config.json (HuggingFace)
- sionic-ai/qwen3-moe-analyzer (GitHub)
- Qwen3 Technical Report (arXiv:2505.09388)
