"""
Layer 0 numerical validation: compare our C engine vs Python reference.
Reads split GGUF, dequantizes weights, computes one DeltaNet layer step.
Compare with C engine's debug output for token 0.
"""
import numpy as np
import struct, os, sys

GGUF_DIR = "D:/models/qwen35-397b-q4/Q4_K_M"
GGUF_BASE = "Qwen3.5-397B-A17B-Q4_K_M"
TOKEN_ID = 151644  # <|im_start|>

def dequant_q8_0_row(data, offset, n_elements):
    """Dequantize one row of Q8_0: 34 bytes per 32 elements."""
    result = np.zeros(n_elements, dtype=np.float64)
    for b in range(n_elements // 32):
        off = offset + b * 34
        d = np.frombuffer(data[off:off+2], dtype=np.float16)[0].astype(np.float64)
        qs = np.frombuffer(data[off+2:off+34], dtype=np.int8).astype(np.float64)
        result[b*32:(b+1)*32] = d * qs
    return result

def find_tensor_in_shards(tensor_name):
    """Find a tensor across split GGUF shards. Returns (shard_path, data_start, tensor_info)."""
    for shard_idx in range(1, 7):
        shard_path = os.path.join(GGUF_DIR, f"{GGUF_BASE}-{shard_idx:05d}-of-00006.gguf")
        if not os.path.exists(shard_path):
            continue
        with open(shard_path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF': continue
            version = struct.unpack('<I', f.read(4))[0]
            n_tensors = struct.unpack('<Q', f.read(8))[0]
            n_kv = struct.unpack('<Q', f.read(8))[0]

            # Skip KV pairs
            for _ in range(n_kv):
                klen = struct.unpack('<Q', f.read(8))[0]
                f.read(klen)
                vtype = struct.unpack('<I', f.read(4))[0]
                if vtype == 4: f.read(4)
                elif vtype == 5: f.read(4)
                elif vtype == 6: f.read(4)
                elif vtype == 8:
                    slen = struct.unpack('<Q', f.read(8))[0]
                    f.read(slen)
                elif vtype == 9:
                    atype = struct.unpack('<I', f.read(4))[0]
                    alen = struct.unpack('<Q', f.read(8))[0]
                    sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:8,10:1}
                    if atype == 8:
                        for _ in range(alen):
                            sl = struct.unpack('<Q', f.read(8))[0]
                            f.read(sl)
                    elif atype in sizes:
                        f.read(alen * sizes[atype])
                    else:
                        f.read(alen * 8)
                elif vtype == 2: f.read(2)
                elif vtype == 7: f.read(8)
                else: f.read(4)

            # Parse tensors
            for _ in range(n_tensors):
                tlen = struct.unpack('<Q', f.read(8))[0]
                tname = f.read(tlen).decode('utf-8')
                n_dims = struct.unpack('<I', f.read(4))[0]
                dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
                ttype = struct.unpack('<I', f.read(4))[0]
                toffset = struct.unpack('<Q', f.read(8))[0]

                if tname == tensor_name:
                    # Data starts at next alignment boundary
                    data_start = ((f.tell() + 31) // 32) * 32
                    # But we need to skip remaining tensor infos first
                    # Actually for split GGUF, data_start is after ALL tensor infos
                    # Let me just compute it properly
                    pass

            # Re-parse to get data_start (after all tensor headers)
            f.seek(0)
            f.read(4)  # magic
            f.read(4)  # version
            n_t = struct.unpack('<Q', f.read(8))[0]
            n_k = struct.unpack('<Q', f.read(8))[0]

            # Skip KV
            for _ in range(n_k):
                klen = struct.unpack('<Q', f.read(8))[0]
                f.read(klen)
                vtype = struct.unpack('<I', f.read(4))[0]
                if vtype == 4: f.read(4)
                elif vtype == 5: f.read(4)
                elif vtype == 6: f.read(4)
                elif vtype == 8:
                    slen = struct.unpack('<Q', f.read(8))[0]
                    f.read(slen)
                elif vtype == 9:
                    atype = struct.unpack('<I', f.read(4))[0]
                    alen = struct.unpack('<Q', f.read(8))[0]
                    sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:8,10:1}
                    if atype == 8:
                        for _ in range(alen):
                            sl = struct.unpack('<Q', f.read(8))[0]
                            f.read(sl)
                    elif atype in sizes:
                        f.read(alen * sizes[atype])
                    else:
                        f.read(alen * 8)
                elif vtype == 2: f.read(2)
                elif vtype == 7: f.read(8)
                else: f.read(4)

            found_tensor = None
            for _ in range(n_t):
                tlen = struct.unpack('<Q', f.read(8))[0]
                tname = f.read(tlen).decode('utf-8')
                n_dims = struct.unpack('<I', f.read(4))[0]
                dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
                ttype = struct.unpack('<I', f.read(4))[0]
                toffset = struct.unpack('<Q', f.read(8))[0]
                if tname == tensor_name:
                    found_tensor = {'dims': dims, 'type': ttype, 'offset': toffset}

            if found_tensor:
                data_start = ((f.tell() + 31) // 32) * 32
                return shard_path, data_start, found_tensor

    return None, None, None

def read_tensor_raw(shard_path, data_start, tensor_info, max_bytes=None):
    """Read raw tensor bytes."""
    with open(shard_path, 'rb') as f:
        f.seek(data_start + tensor_info['offset'])
        dims = tensor_info['dims']
        n_elem = 1
        for d in dims: n_elem *= d
        if tensor_info['type'] == 8:  # Q8_0
            nbytes = (n_elem // 32) * 34
        elif tensor_info['type'] == 0:  # F32
            nbytes = n_elem * 4
        else:
            nbytes = n_elem * 4  # guess
        if max_bytes: nbytes = min(nbytes, max_bytes)
        return f.read(nbytes), n_elem

if __name__ == '__main__':
    H = 4096

    # 1. Read embedding for token 151644
    print("=== Reading embedding ===")
    path, ds, info = find_tensor_in_shards("token_embd.weight")
    if not info:
        print("ERROR: token_embd.weight not found!")
        sys.exit(1)
    print(f"  Found in {os.path.basename(path)}, dims={info['dims']}, type={info['type']}")

    row_bytes = (H // 32) * 34  # Q8_0 bytes per row
    with open(path, 'rb') as f:
        f.seek(ds + info['offset'] + TOKEN_ID * row_bytes)
        embd_raw = f.read(row_bytes)
    embedding = dequant_q8_0_row(embd_raw, 0, H)
    print(f"  embedding[0..3]: {embedding[0]:.6f} {embedding[1]:.6f} {embedding[2]:.6f} {embedding[3]:.6f}")
    print(f"  C engine shows:  0.011051 0.020333 -0.004862 -0.021217")

    # 2. Read attn_norm for layer 0
    print("\n=== Reading attn_norm.weight ===")
    path, ds, info = find_tensor_in_shards("blk.0.attn_norm.weight")
    if not info:
        print("ERROR: not found!")
        sys.exit(1)
    with open(path, 'rb') as f:
        f.seek(ds + info['offset'])
        norm_w = np.frombuffer(f.read(H * 4), dtype=np.float32).astype(np.float64)
    print(f"  norm_w[0..3]: {norm_w[0]:.6f} {norm_w[1]:.6f} {norm_w[2]:.6f} {norm_w[3]:.6f}")

    # 3. Apply RMSNorm (double precision)
    rms = np.sqrt(np.mean(embedding ** 2) + 1e-6)
    normed = (embedding / rms) * norm_w
    print(f"\n=== RMSNorm result ===")
    print(f"  rms = {rms:.6f}")
    print(f"  normed[0..7]: {normed[0]:.6f} {normed[1]:.6f} {normed[2]:.6f} {normed[3]:.6f} {normed[4]:.6f} {normed[5]:.6f} {normed[6]:.6f} {normed[7]:.6f}")
    print(f"  C engine shows: 0.888368 1.607719 -0.402783 -1.721444 -0.690243 -0.227780 0.668531 0.622534")

    # Check match
    expected = [0.888368, 1.607719, -0.402783, -1.721444, -0.690243, -0.227780, 0.668531, 0.622534]
    max_err = max(abs(normed[i] - expected[i]) for i in range(8))
    print(f"  Max error vs C engine: {max_err:.6f}")
    if max_err < 0.01:
        print("  OK: RMSNorm MATCHES!")
    else:
        print(f"  FAIL: MISMATCH! (error = {max_err:.6f})")

    # 4. Read QKV weight for layer 0 and compute projection
    print("\n=== QKV projection ===")
    path, ds, info = find_tensor_in_shards("blk.0.attn_qkv.weight")
    if info:
        print(f"  Found in {os.path.basename(path)}, dims={info['dims']}, type={info['type']}")
        QKV_DIM = 12288
        # Read first few rows to check
        raw, n_elem = read_tensor_raw(path, ds, info)
        print(f"  Read {len(raw)} bytes ({n_elem} elements)")

        # Compute QKV = W @ normed (Q8_0 weight × FP64 input)
        normed_f32 = normed.astype(np.float32)
        qkv = np.zeros(QKV_DIM, dtype=np.float64)
        for row in range(min(8, QKV_DIM)):  # just first 8 rows for comparison
            row_vec = dequant_q8_0_row(raw, row * (H // 32) * 34, H)
            qkv[row] = np.dot(row_vec, embedding / rms * norm_w)

        print(f"  qkv[0..7] (Python FP64): {qkv[0]:.6f} {qkv[1]:.6f} {qkv[2]:.6f} {qkv[3]:.6f} {qkv[4]:.6f} {qkv[5]:.6f} {qkv[6]:.6f} {qkv[7]:.6f}")
        print(f"  C engine qkv_pre_conv:    0.787797 -0.701947 -4.064668 -0.181013 0.621780 -0.173210 0.914931 1.411521")

        qkv_expected = [0.787797, -0.701947, -4.064668, -0.181013, 0.621780, -0.173210, 0.914931, 1.411521]
        max_err_qkv = max(abs(qkv[i] - qkv_expected[i]) for i in range(8))
        print(f"  Max error vs C engine: {max_err_qkv:.6f}")
        if max_err_qkv < 0.1:
            print("  OK: QKV projection MATCHES!")
        else:
            print(f"  FAIL: QKV MISMATCH! This could be the bug source.")
    else:
        print("  QKV weight not found in any shard")

    print("\nDone.")
