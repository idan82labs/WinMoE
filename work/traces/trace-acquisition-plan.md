# Trace Acquisition Plan

## Target: Real routing traces from Qwen3-30B-A3B

### Why Qwen3-30B-A3B
- 128 experts, top-8, 48 layers, no shared experts
- Q4_K_M GGUF = 18.6 GB → fits in 40 GB RAM
- HuggingFace 4-bit = ~18 GB
- Same model tested in FlashMoE paper (arXiv 2601.17063) and PreScope

### Method A: HuggingFace transformers + bitsandbytes
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B",
    load_in_4bit=True,
    device_map="auto"
)
# Forward hooks on mlp.gate for each layer
```
- Captures: router_logits, expert_indices, gate_weights per token per layer
- Limitation: slow inference; bitsandbytes Windows support may be unstable

### Method B: llama.cpp Q4_K_M + separate HF trace pass
- Run inference via llama.cpp for speed
- Capture traces separately via HF forward pass on same prompts

### Scaling to Qwen3.5-397B-A17B
- Use Qwen3-30B trace statistics to calibrate Zipf parameters
- Model 397B routing as scaled version (512 experts, top-10, 60 layers)
- Validate by checking if 30B Zipf fits transfer to 397B structure

### Workloads to trace
- General chat (ShareGPT dataset samples)
- Code generation
- Math reasoning
- Short prompts (<50 tokens) and long prompts (500+ tokens)

### Required outputs per trace record
- workload_id, token_index, layer_index
- active_expert_ids (top-K)
- gate_weights (optional but valuable)
- timestamp or ordering

### Fallback: Use synthetic traces calibrated by literature
- Zipf s=1.2 (confirmed by arXiv:2511.05814)
- Layer-varying skew (input/output layers more skewed)
- Adjacent-token overlap via Markov model
