"""
Flash-MoE V2 — Real Routing Trace Capture from Qwen3-30B-A3B

Captures per-token, per-layer expert routing decisions using HuggingFace
transformers forward hooks.

Requires: ~18 GB RAM (4-bit quantization), or ~32 GB (8-bit)

Usage:
  python capture_traces.py --model Qwen/Qwen3-30B-A3B --prompt "Hello world" --output traces.npz
  python capture_traces.py --model Qwen/Qwen3-30B-A3B --dataset shareGPT --max-tokens 1000
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Requires: pip install torch transformers accelerate bitsandbytes")
    sys.exit(1)


class RoutingTraceCapture:
    """Captures MoE routing decisions via forward hooks."""

    def __init__(self):
        self.traces = []  # list of per-forward-pass records
        self._current_record = {}
        self._hooks = []

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # Qwen3MoeTopKRouter returns (router_logits, routing_weights, selected_experts)
            # router_logits: [batch*seq, num_experts]
            # routing_weights: [batch*seq, top_k]
            # selected_experts: [batch*seq, top_k]
            if isinstance(output, tuple) and len(output) == 3:
                logits, weights, indices = output
            elif isinstance(output, tuple) and len(output) == 2:
                # Some versions return (logits, indices)
                logits, indices = output
                weights = None
            else:
                return

            self._current_record[layer_idx] = {
                "expert_ids": indices.detach().cpu().numpy(),
                "gate_weights": weights.detach().cpu().numpy() if weights is not None else None,
                "router_logits": logits.detach().cpu().numpy() if logits.shape[-1] <= 512 else None,
            }
        return hook_fn

    def attach(self, model):
        """Attach forward hooks to all MoE gate modules."""
        layer_idx = 0
        for name, module in model.named_modules():
            # Qwen3MoE uses module named 'gate' inside the MoE block
            if name.endswith(".gate") and hasattr(module, "weight"):
                hook = module.register_forward_hook(self._make_hook(layer_idx))
                self._hooks.append(hook)
                layer_idx += 1
        print(f"Attached hooks to {layer_idx} MoE layers")
        return layer_idx

    def start_record(self):
        self._current_record = {}

    def finish_record(self):
        if self._current_record:
            self.traces.append(dict(self._current_record))
            self._current_record = {}

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def save(self, path: str):
        """Save traces as compressed npz."""
        if not self.traces:
            print("No traces to save")
            return

        # Convert to structured arrays
        num_records = len(self.traces)
        num_layers = len(self.traces[0])
        sample = next(iter(self.traces[0].values()))
        K = sample["expert_ids"].shape[-1]

        print(f"Saving {num_records} trace records, {num_layers} layers, K={K}")

        # Stack into (tokens, layers, K) array
        # Note: each record may have multiple tokens (seq_len > 1 during prefill)
        all_expert_ids = []
        all_gate_weights = []

        for record in self.traces:
            layer_ids = []
            layer_weights = []
            for l in range(num_layers):
                if l in record:
                    layer_ids.append(record[l]["expert_ids"])
                    if record[l]["gate_weights"] is not None:
                        layer_weights.append(record[l]["gate_weights"])
            if layer_ids:
                # Shape: (seq_len, K) per layer → stack to (num_layers, seq_len, K)
                stacked_ids = np.stack(layer_ids, axis=0)  # (layers, seq, K)
                all_expert_ids.append(stacked_ids)
                if layer_weights:
                    stacked_w = np.stack(layer_weights, axis=0)
                    all_gate_weights.append(stacked_w)

        # Concatenate along token dimension
        # Final shape: (layers, total_tokens, K) → transpose to (total_tokens, layers, K)
        ids_arr = np.concatenate(all_expert_ids, axis=1)  # (layers, total_tokens, K)
        ids_arr = ids_arr.transpose(1, 0, 2)  # (total_tokens, layers, K)

        save_dict = {"traces": ids_arr}
        if all_gate_weights:
            w_arr = np.concatenate(all_gate_weights, axis=1).transpose(1, 0, 2)
            save_dict["gate_weights"] = w_arr

        np.savez_compressed(path, **save_dict)
        print(f"Saved traces to {path}: shape={ids_arr.shape}")


def capture_from_prompts(model_name: str, prompts: list, output_path: str,
                          load_in_4bit: bool = True, max_new_tokens: int = 50):
    """Capture routing traces from a list of prompts."""
    print(f"Loading model: {model_name}")
    print(f"Quantization: {'4-bit' if load_in_4bit else 'float16'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    load_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    # Attach trace capture
    capture = RoutingTraceCapture()
    num_layers = capture.attach(model)

    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}/{len(prompts)}: {prompt[:80]}...")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate tokens one at a time to get per-token traces
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            for step in range(max_new_tokens):
                capture.start_record()
                outputs = model(input_ids)
                capture.finish_record()

                # Get next token
                next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                # Stop on EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break

        print(f"  Generated {len(capture.traces)} records")

    # Save
    capture.save(output_path)
    capture.detach()


def main():
    parser = argparse.ArgumentParser(description="Capture MoE routing traces")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to trace")
    parser.add_argument("--prompts-file", type=str, default=None,
                        help="JSON file with list of prompts")
    parser.add_argument("--output", type=str, default="traces.npz")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization")
    args = parser.parse_args()

    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = json.load(f)
    else:
        prompts = [
            "Explain the theory of relativity in simple terms.",
            "Write a Python function to sort a list using merge sort.",
            "What is the capital of France and why is it important?",
            "Solve the equation: 3x + 5 = 20",
            "Tell me a short story about a robot learning to cook.",
        ]

    capture_from_prompts(
        args.model, prompts, args.output,
        load_in_4bit=not args.no_4bit,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
