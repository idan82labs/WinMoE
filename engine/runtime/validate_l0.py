"""Quick L0 validation: load weights from GGUF, compute first layer, compare."""
import struct, numpy as np, sys

def read_q8_0_block(data, offset):
    """Dequant one Q8_0 block (32 weights): 2-byte FP16 scale + 32 int8 values."""
    d_bytes = data[offset:offset+2]
    d = np.frombuffer(d_bytes, dtype=np.float16)[0].astype(np.float32)
    qs = np.frombuffer(data[offset+2:offset+34], dtype=np.int8).astype(np.float32)
    return d * qs

def dequant_q8_0(data, rows, cols):
    """Dequant full Q8_0 tensor [rows, cols]."""
    blocks_per_row = cols // 32
    result = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for b in range(blocks_per_row):
            offset = (r * blocks_per_row + b) * 34
            result[r, b*32:(b+1)*32] = read_q8_0_block(data, offset)
    return result

def rmsnorm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x**2) + eps)
    return weight * x / rms

# Our engine's L0 embedding output for token 151644 (im_start)
# Let's just check the RMSNorm and first matmul
print("Loading L0 norm weights from our engine's output...")
print("Our engine L0 after attn: h[0..3]=-0.1717 -0.4027 -0.5814 0.4934")
print("Our engine L0 after MoE:  h[0..3]=-0.1673 -0.4010 -0.5800 0.4939")

# Key test: does our embedding for token 151644 match?
# Our engine prints: Embedding[0..3]: -0.000223 0.000223 0.003346 0.020074
# This is for the INITIAL prompt token. With the chat prompt,
# the first token is 151644 (<|im_start|>).
print("\nOur engine embedding[0..3]: -0.000223 0.000223 0.003346 0.020074")
print("(This should be the embedding for token 151644)")
print("\nTo verify: need to dequant token 151644's embedding from GGUF")
print("and compare against our engine's output.")
print("\nBut the fastest verification: compare our FIRST GENERATED TOKEN's")
print("logit argmax against llama.cpp's first token (151644 for thinking)")
print("Our engine produces token 6948 or 23092 (gibberish)")
print("llama.cpp produces token 151644 (<|im_start|> thinking)")
print("\n==> The computation diverges. Need layer-by-layer comparison.")
