"""
Flash-MoE Windows V2 — Run Qwen3.5-397B-A17B

Uses llama-cpp-python with the GGUF model.
Streams expert weights from SSD with partial GPU offload.

Usage:
  python run_397b.py [--prompt "Hello"] [--n-gpu-layers 10] [--ctx 2048]
"""
import argparse
import os
import sys
import time
from glob import glob

def find_gguf_files(cache_dir="D:/hf_cache"):
    """Find the split GGUF files for Qwen3.5-397B."""
    patterns = [
        os.path.join(cache_dir, "models--unsloth--Qwen3.5-397B-A17B-GGUF",
                     "snapshots", "*", "UD-IQ2_XXS", "*.gguf"),
        os.path.join(cache_dir, "models--unsloth--Qwen3.5-397B-A17B-GGUF",
                     "snapshots", "*", "Q4_K_S", "*.gguf"),
    ]
    for pattern in patterns:
        files = sorted(glob(pattern))
        if files:
            return files
    # Try blobs
    blobs = os.path.join(cache_dir, "models--unsloth--Qwen3.5-397B-A17B-GGUF", "blobs")
    if os.path.exists(blobs):
        files = sorted([os.path.join(blobs, f) for f in os.listdir(blobs)
                        if os.path.getsize(os.path.join(blobs, f)) > 1e9])
        if files:
            return files
    return []


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3.5-397B")
    parser.add_argument("--prompt", type=str,
                        default="Explain quantum computing in simple terms.")
    parser.add_argument("--n-gpu-layers", type=int, default=5,
                        help="Number of layers to offload to GPU (default 5)")
    parser.add_argument("--ctx", type=int, default=2048,
                        help="Context length")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Max tokens to generate")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Direct path to GGUF file (first shard for split models)")
    args = parser.parse_args()

    from llama_cpp import Llama

    # Find model
    if args.model_path:
        model_path = args.model_path
    else:
        files = find_gguf_files()
        if not files:
            print("No GGUF files found! Download the model first.")
            sys.exit(1)
        # Use the first shard (llama.cpp handles split files automatically)
        model_path = files[0]

    print("=" * 60)
    print("Flash-MoE Windows V2 -- Qwen3.5-397B Inference")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"GPU layers: {args.n_gpu_layers}")
    print(f"Context: {args.ctx}")
    print()

    # Load model
    print("Loading model...")
    t0 = time.time()
    model = Llama(
        model_path=model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.ctx,
        verbose=True,
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Generate
    print(f"\nPrompt: {args.prompt}")
    print(f"Generating {args.max_tokens} tokens...\n")
    print("-" * 40)

    t_start = time.time()
    output = model.create_completion(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=0.7,
        top_p=0.9,
        stream=True,
    )

    tokens_generated = 0
    first_token_time = None
    for chunk in output:
        text = chunk["choices"][0]["text"]
        if text:
            if first_token_time is None:
                first_token_time = time.time()
            print(text, end="", flush=True)
            tokens_generated += 1

    t_end = time.time()
    print("\n" + "-" * 40)

    # Stats
    total_time = t_end - t_start
    ttft = first_token_time - t_start if first_token_time else 0
    decode_time = t_end - first_token_time if first_token_time else total_time
    tps = tokens_generated / decode_time if decode_time > 0 else 0

    print(f"\nTokens: {tokens_generated}")
    print(f"Time to first token: {ttft:.1f}s")
    print(f"Decode time: {decode_time:.1f}s")
    print(f"Speed: {tps:.2f} tok/s")
    print(f"Total: {total_time:.1f}s")


if __name__ == "__main__":
    main()
