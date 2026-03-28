"""
Validate L0 computation: embedding → RMSNorm → QKV matmul
Compare against WinMoE engine output to find where divergence starts.
"""
import struct, sys, os
import numpy as np

SHARD_PATH = "D:/models/qwen35-397b-q4/Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-00001-of-00006.gguf"
TOKEN_ID = 151644  # <|im_start|>
HIDDEN = 4096

def read_u32(f): return struct.unpack("<I", f.read(4))[0]
def read_u64(f): return struct.unpack("<Q", f.read(8))[0]
def read_i32(f): return struct.unpack("<i", f.read(4))[0]

def skip_gguf_value(f, vtype):
    sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:4, 8:8, 10:8, 12:8}
    if vtype in sizes:
        f.read(sizes[vtype])
    elif vtype == 4:  # string
        slen = read_u64(f)
        f.read(slen)
    elif vtype == 9:  # array
        arr_type = read_u32(f)
        arr_len = read_u64(f)
        for _ in range(arr_len):
            skip_gguf_value(f, arr_type)
    else:
        print(f"Unknown vtype {vtype}")
        sys.exit(1)

def dequant_q8_0_row(data, offset, n_elements):
    """Dequant one Q8_0 row: (n_elements/32) blocks of 34 bytes each."""
    blocks = n_elements // 32
    result = np.zeros(n_elements, dtype=np.float32)
    for b in range(blocks):
        bo = offset + b * 34
        d = np.frombuffer(data[bo:bo+2], dtype=np.float16)[0].astype(np.float32)
        qs = np.frombuffer(data[bo+2:bo+34], dtype=np.int8).astype(np.float32)
        result[b*32:(b+1)*32] = d * qs
    return result

def rmsnorm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x**2) + eps)
    return weight * x / rms

# Parse GGUF header to find tensor info
print("Parsing GGUF header...")
with open(SHARD_PATH, "rb") as f:
    magic = f.read(4)
    version = read_u32(f)
    n_tensors = read_u64(f)
    n_kv = read_u64(f)
    print(f"  GGUF v{version}, {n_tensors} tensors, {n_kv} KV pairs")

    # Skip KV pairs
    for _ in range(n_kv):
        key_len = read_u64(f)
        key = f.read(key_len).decode('utf-8', errors='ignore')
        vtype = read_u32(f)
        skip_gguf_value(f, vtype)

    # Read tensor info
    tensors = {}
    for ti in range(n_tensors):
        name_len = read_u64(f)
        name = f.read(name_len).decode('utf-8')
        n_dims = read_u32(f)
        dims = [read_u64(f) for _ in range(n_dims)]
        ttype = read_u32(f)
        offset = read_u64(f)
        tensors[name] = {'dims': dims, 'type': ttype, 'offset': offset}

    # Data section starts aligned
    data_start = ((f.tell() + 31) // 32) * 32

    # Find embedding tensor
    embd_info = tensors.get('token_embd.weight')
    norm_info = tensors.get('blk.0.attn_norm.weight')

    if not embd_info or not norm_info:
        print("Missing tensors!")
        sys.exit(1)

    print(f"  token_embd: dims={embd_info['dims']}, type={embd_info['type']}")
    print(f"  L0 attn_norm: dims={norm_info['dims']}, type={norm_info['type']}")

    # Read embedding for TOKEN_ID (Q8_0)
    row_bytes = (HIDDEN // 32) * 34  # 4352 bytes per token
    embd_abs = data_start + embd_info['offset'] + TOKEN_ID * row_bytes
    f.seek(embd_abs)
    embd_raw = f.read(row_bytes)
    embedding = dequant_q8_0_row(embd_raw, 0, HIDDEN)
    print(f"\nEmbedding for token {TOKEN_ID}:")
    print(f"  [0..7]: {embedding[:8]}")
    print(f"  rms: {np.sqrt(np.mean(embedding**2)):.6f}")

    # Read L0 attn_norm (FP32)
    norm_abs = data_start + norm_info['offset']
    f.seek(norm_abs)
    norm_data = f.read(HIDDEN * 4)
    norm_weight = np.frombuffer(norm_data, dtype=np.float32)
    print(f"\nL0 attn_norm weight [0..7]: {norm_weight[:8]}")

    # Apply RMSNorm
    normed = rmsnorm(embedding, norm_weight)
    print(f"\nNormed [0..7]: {normed[:8]}")
    print(f"  rms: {np.sqrt(np.mean(normed**2)):.6f}")

    # Also read QKV weight and compute first output value
    qkv_info = tensors.get('blk.0.attn_qkv.weight')
    if qkv_info:
        print(f"\nL0 QKV: dims={qkv_info['dims']}, type={qkv_info['type']}")
        qkv_abs = data_start + qkv_info['offset']
        # Read first row only (one QKV output element)
        f.seek(qkv_abs)
        qkv_row_bytes = (HIDDEN // 32) * 34
        qkv_row_raw = f.read(qkv_row_bytes)
        qkv_row0 = dequant_q8_0_row(qkv_row_raw, 0, HIDDEN)
        # Compute dot product: qkv_out[0] = dot(qkv_row0, normed)
        qkv_out0 = np.dot(qkv_row0, normed)
        print(f"  QKV row 0 [0..7]: {qkv_row0[:8]}")
        print(f"  QKV output[0] = {qkv_out0:.6f}")

    print("\n=== Compare these values against WinMoE engine debug output ===")
    print("Add to winmoe_inference.c: print hidden[0..7] after embedding,")
    print("normed[0..7] after RMSNorm, and qkv[0] after QKV projection.")
