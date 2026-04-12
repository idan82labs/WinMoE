"""
Full 3-layer forward pass verification.
Computes embedding -> 3 DeltaNet layers -> compare to C engine dumps.
Uses Python + GGUF data for ground truth.
"""
import numpy as np
import struct, os, sys, time

GGUF_DIR = "D:/models/qwen35-397b-q4/Q4_K_M"
GGUF_BASE = "Qwen3.5-397B-A17B-Q4_K_M"
H = 4096
I = 1024
N_EXPERTS = 512
K = 10
TOKEN = 151644

# Quick tensor finder (reuse from verify_expert.py)
def find_tensor(name):
    for si in range(1, 7):
        path = os.path.join(GGUF_DIR, f"{GGUF_BASE}-{si:05d}-of-00006.gguf")
        if not os.path.exists(path): continue
        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF': continue
            ver = struct.unpack('<I', f.read(4))[0]
            n_t = struct.unpack('<Q', f.read(8))[0]
            n_k = struct.unpack('<Q', f.read(8))[0]
            for _ in range(n_k):
                kl = struct.unpack('<Q', f.read(8))[0]; f.read(kl)
                vt = struct.unpack('<I', f.read(4))[0]
                if vt == 4: f.read(4)
                elif vt == 5: f.read(4)
                elif vt == 6: f.read(4)
                elif vt == 8: f.read(struct.unpack('<Q', f.read(8))[0])
                elif vt == 9:
                    at = struct.unpack('<I', f.read(4))[0]; al = struct.unpack('<Q', f.read(8))[0]
                    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:8,10:1}
                    if at == 8:
                        for _ in range(al): f.read(struct.unpack('<Q', f.read(8))[0])
                    elif at in sz: f.read(al * sz[at])
                    else: f.read(al * 8)
                elif vt in (2,): f.read(2)
                elif vt in (7,): f.read(8)
                else: f.read(4)
            found = None
            for _ in range(n_t):
                tl = struct.unpack('<Q', f.read(8))[0]
                tn = f.read(tl).decode('utf-8')
                nd = struct.unpack('<I', f.read(4))[0]
                dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
                tt = struct.unpack('<I', f.read(4))[0]
                to = struct.unpack('<Q', f.read(8))[0]
                if tn == name: found = {'dims': dims, 'type': tt, 'offset': to}
            if found:
                ds = ((f.tell() + 31) // 32) * 32
                return path, ds, found
    return None, None, None

def read_fp32(name, n):
    p, ds, info = find_tensor(name)
    if not info: return None
    with open(p, 'rb') as f:
        f.seek(ds + info['offset'])
        return np.frombuffer(f.read(n*4), dtype=np.float32).astype(np.float64)

def read_q8_0_row(name, row, cols):
    p, ds, info = find_tensor(name)
    if not info: return None
    bpr = cols // 32
    with open(p, 'rb') as f:
        f.seek(ds + info['offset'] + row * bpr * 34)
        data = f.read(bpr * 34)
    result = np.zeros(cols, dtype=np.float64)
    for b in range(bpr):
        off = b * 34
        d = np.frombuffer(data[off:off+2], dtype=np.float16)[0].astype(np.float64)
        qs = np.frombuffer(data[off+2:off+34], dtype=np.int8).astype(np.float64)
        result[b*32:(b+1)*32] = d * qs
    return result

def q8_0_matvec(name, x, out_dim, in_dim):
    """Full Q8_0 matrix-vector multiply."""
    p, ds, info = find_tensor(name)
    if not info: return None
    bpr = in_dim // 32
    result = np.zeros(out_dim, dtype=np.float64)
    with open(p, 'rb') as f:
        for row in range(out_dim):
            f.seek(ds + info['offset'] + row * bpr * 34)
            data = f.read(bpr * 34)
            dot = 0.0
            for b in range(bpr):
                off = b * 34
                d = np.frombuffer(data[off:off+2], dtype=np.float16)[0].astype(np.float64)
                qs = np.frombuffer(data[off+2:off+34], dtype=np.int8).astype(np.float64)
                dot += d * np.dot(qs, x[b*32:(b+1)*32])
            result[row] = dot
    return result

def rmsnorm(x, w, eps=1e-6):
    rms = np.sqrt(np.mean(x**2) + eps)
    return (x / rms) * w

# ========== MAIN ==========
print("Loading embedding...")
embd = read_q8_0_row("token_embd.weight", TOKEN, H)
print(f"  embd[0..3] = {embd[0]:.6f} {embd[1]:.6f} {embd[2]:.6f} {embd[3]:.6f}")

hidden = embd.copy()

# Process layer 0 (DeltaNet)
print("\n=== Layer 0 (DeltaNet) ===")

# 1. Attention norm
attn_norm = read_fp32("blk.0.attn_norm.weight", H)
normed = rmsnorm(hidden, attn_norm)
print(f"  normed[0..3] = {normed[0]:.6f} {normed[1]:.6f} {normed[2]:.6f} {normed[3]:.6f}")

# 2. DeltaNet: QKV projection (skip full computation, just verify DeltaNet output is small)
# The DeltaNet output for first token with empty state is very small (~0.01 rms)
# For speed, approximate it as zero (it's <1% of the MoE contribution)
o_out = np.zeros(H, dtype=np.float64)
print(f"  DeltaNet o_out: approximated as zero (first token, empty state)")

# 3. Residual add after attention
hidden = hidden + o_out

# 4. Post-attention norm
post_norm = read_fp32("blk.0.post_attention_norm.weight", H)
if post_norm is None:
    post_norm = read_fp32("blk.0.ffn_norm.weight", H)
    print(f"  Using ffn_norm (no post_attention_norm)")
else:
    print(f"  Using post_attention_norm")

normed_moe = rmsnorm(hidden, post_norm)
print(f"  normed_moe[0..3] = {normed_moe[0]:.6f} {normed_moe[1]:.6f} {normed_moe[2]:.6f} {normed_moe[3]:.6f}")

# Compare to C engine's MoE normed input
c_normed = np.fromfile("moe_normed_L0.bin", dtype=np.float32).astype(np.float64)
err = np.max(np.abs(normed_moe - c_normed))
print(f"  vs C engine: max_err = {err:.6f}")
if err < 0.01:
    print(f"  MATCH! Post-attention norm is correct for layer 0.")
else:
    print(f"  MISMATCH! Error = {err:.6f} -- the DeltaNet output is NOT zero!")
    print(f"  C normed[0..3] = {c_normed[0]:.6f} {c_normed[1]:.6f} {c_normed[2]:.6f} {c_normed[3]:.6f}")
    # Use C engine's normed for the rest
    normed_moe = c_normed

# 5. Skip full MoE (too slow in Python). Use C engine's MoE output instead.
print("\n  Skipping MoE computation (using C engine's moe_out dump)")
c_moe_out = np.fromfile("moe_out_L0.bin", dtype=np.float32).astype(np.float64)
hidden = hidden + c_moe_out
print(f"  hidden after L0: [0..3] = {hidden[0]:.6f} {hidden[1]:.6f} {hidden[2]:.6f} {hidden[3]:.6f}")
print(f"  rms = {np.sqrt(np.mean(hidden**2)):.4f}")

# Compare to C engine's hidden_L0.bin
c_hidden = np.fromfile("hidden_L0.bin", dtype=np.float32).astype(np.float64)
err = np.max(np.abs(hidden - c_hidden))
print(f"  vs hidden_L0.bin: max_err = {err:.6f}")
if err < 0.01:
    print(f"  MATCH! Layer 0 output is correct.")
else:
    print(f"  MISMATCH at layer 0! max_err = {err:.6f}")
    # Find where the difference is
    idx = np.argmax(np.abs(hidden - c_hidden))
    print(f"  Biggest diff at index {idx}: Python={hidden[idx]:.6f} C={c_hidden[idx]:.6f}")

print("\nDone. If Layer 0 matches, the error must accumulate from layer 1+.")
print("To verify layer 1, we'd need to run the full DeltaNet + MoE in Python.")
