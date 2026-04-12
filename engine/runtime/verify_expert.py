"""Verify expert 213 gate FFN output for layer 0."""
import numpy as np
import struct, os

GGUF_DIR = "D:/models/qwen35-397b-q4/Q4_K_M"
GGUF_BASE = "Qwen3.5-397B-A17B-Q4_K_M"
H = 4096
I = 1024
N_EXPERTS = 512
EXPERT_ID = 213

def find_tensor(name):
    for shard_idx in range(1, 7):
        path = os.path.join(GGUF_DIR, f"{GGUF_BASE}-{shard_idx:05d}-of-00006.gguf")
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
                    at = struct.unpack('<I', f.read(4))[0]
                    al = struct.unpack('<Q', f.read(8))[0]
                    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:8,10:1}
                    if at == 8:
                        for _ in range(al): f.read(struct.unpack('<Q', f.read(8))[0])
                    elif at in sz: f.read(al * sz[at])
                    else: f.read(al * 8)
                elif vt == 2: f.read(2)
                elif vt == 7: f.read(8)
                else: f.read(4)
            found = None
            for _ in range(n_t):
                tl = struct.unpack('<Q', f.read(8))[0]
                tn = f.read(tl).decode('utf-8')
                nd = struct.unpack('<I', f.read(4))[0]
                dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
                tt = struct.unpack('<I', f.read(4))[0]
                to = struct.unpack('<Q', f.read(8))[0]
                if tn == name:
                    found = {'dims': dims, 'type': tt, 'offset': to}
            if found:
                ds = ((f.tell() + 31) // 32) * 32
                return path, ds, found
    return None, None, None

def dequant_q4k_block(data, offset):
    """Dequantize one Q4_K block (144 bytes -> 256 float values)."""
    d = np.frombuffer(data[offset:offset+2], dtype=np.float16)[0].astype(np.float64)
    dmin = np.frombuffer(data[offset+2:offset+4], dtype=np.float16)[0].astype(np.float64)
    sc_bytes = data[offset+4:offset+16]
    qs = data[offset+16:offset+144]

    # Decode 6-bit scales and mins (GGML get_scale_min_k4)
    scales = np.zeros(8, dtype=np.int32)
    mins = np.zeros(8, dtype=np.int32)
    scales[0] = sc_bytes[0] & 63; mins[0] = sc_bytes[4] & 63
    scales[1] = sc_bytes[1] & 63; mins[1] = sc_bytes[5] & 63
    scales[2] = sc_bytes[2] & 63; mins[2] = sc_bytes[6] & 63
    scales[3] = sc_bytes[3] & 63; mins[3] = sc_bytes[7] & 63
    scales[4] = (sc_bytes[8] & 0xF) | ((sc_bytes[0] >> 6) << 4)
    scales[5] = (sc_bytes[9] & 0xF) | ((sc_bytes[1] >> 6) << 4)
    scales[6] = (sc_bytes[10] & 0xF) | ((sc_bytes[2] >> 6) << 4)
    scales[7] = (sc_bytes[11] & 0xF) | ((sc_bytes[3] >> 6) << 4)
    mins[4] = (sc_bytes[8] >> 4) | ((sc_bytes[4] >> 6) << 4)
    mins[5] = (sc_bytes[9] >> 4) | ((sc_bytes[5] >> 6) << 4)
    mins[6] = (sc_bytes[10] >> 4) | ((sc_bytes[6] >> 6) << 4)
    mins[7] = (sc_bytes[11] >> 4) | ((sc_bytes[7] >> 6) << 4)

    result = np.zeros(256, dtype=np.float64)
    for sub in range(8):
        sc_f = d * float(scales[sub])
        mn_f = dmin * float(mins[sub])
        for j in range(16):
            packed = qs[sub * 16 + j]
            w0 = sc_f * float(packed & 0x0F) - mn_f
            w1 = sc_f * float(packed >> 4) - mn_f
            result[sub * 32 + j] = w0
            result[sub * 32 + j + 16] = w1
    return result

# Read MoE normed input
normed = np.fromfile("moe_normed_L0.bin", dtype=np.float32).astype(np.float64)
print(f"normed rms={np.sqrt(np.mean(normed**2)):.4f}")

# Find gate_exps tensor for layer 0
path, ds, info = find_tensor("blk.0.ffn_gate_exps.weight")
if not info:
    print("gate_exps not found!")
    exit()

print(f"gate_exps: dims={info['dims']} type={info['type']} in {os.path.basename(path)}")

# Q4_K: type=12. bytes per block=144, weights per block=256
blocks_per_row = H // 256  # 16
bytes_per_row = blocks_per_row * 144  # 2304
per_expert = I * bytes_per_row  # 2,359,296
print(f"per_expert={per_expert} bytes_per_row={bytes_per_row}")

# Read expert 213's gate weight
with open(path, 'rb') as f:
    expert_offset = ds + info['offset'] + EXPERT_ID * per_expert
    f.seek(expert_offset)
    expert_data = f.read(per_expert)

print(f"\nExpert {EXPERT_ID} gate weight:")
print(f"  First 16 bytes: {expert_data[:16].hex()}")

# Check first block header
d16 = struct.unpack('<H', expert_data[0:2])[0]
dmin16 = struct.unpack('<H', expert_data[2:4])[0]
print(f"  Block 0: d=0x{d16:04x} dmin=0x{dmin16:04x}")

# Dequant and compute gate output for first 3 rows
gate_out = np.zeros(I, dtype=np.float64)
for row in range(I):
    row_offset = row * bytes_per_row
    row_sum = 0.0
    for b in range(blocks_per_row):
        block_offset = row_offset + b * 144
        weights = dequant_q4k_block(expert_data, block_offset)
        for j in range(256):
            row_sum += weights[j] * normed[b * 256 + j]
    gate_out[row] = row_sum

print(f"\nPython gate output:")
print(f"  gate_rms = {np.sqrt(np.mean(gate_out**2)):.6f}")
print(f"  gate[0..7] = {gate_out[0]:.6f} {gate_out[1]:.6f} {gate_out[2]:.6f} {gate_out[3]:.6f} {gate_out[4]:.6f} {gate_out[5]:.6f} {gate_out[6]:.6f} {gate_out[7]:.6f}")
print(f"  C engine:    gate_rms=0.0445 gate[0..3]=0.0000 0.0000 0.0000 0.0000")

# Also compute UP projection
path_up, ds_up, info_up = find_tensor("blk.0.ffn_up_exps.weight")
if info_up:
    with open(path_up, 'rb') as f:
        f.seek(ds_up + info_up['offset'] + EXPERT_ID * per_expert)
        up_data = f.read(per_expert)

    up_out = np.zeros(I, dtype=np.float64)
    for row in range(I):
        row_offset = row * bytes_per_row
        row_sum = 0.0
        for b in range(blocks_per_row):
            weights = dequant_q4k_block(up_data, row_offset + b * 144)
            for j in range(256):
                row_sum += weights[j] * normed[b * 256 + j]
        up_out[row] = row_sum

    print(f"\nPython up output: rms={np.sqrt(np.mean(up_out**2)):.6f}")
    print(f"  up[0..3] = {up_out[0]:.6f} {up_out[1]:.6f} {up_out[2]:.6f} {up_out[3]:.6f}")

    # SwiGLU
    act = gate_out * (1.0 / (1.0 + np.exp(-gate_out))) * up_out
    print(f"\nSwiGLU act: rms={np.sqrt(np.mean(act**2)):.6f}")
    print(f"  C engine act_rms=0.0050")

    # DOWN projection (Q5_K)
    path_dn, ds_dn, info_dn = find_tensor("blk.0.ffn_down_exps.weight")
    if info_dn:
        print(f"\ndown_exps: dims={info_dn['dims']} type={info_dn['type']}")
        # Q5_K: type=13, 176 bytes per block, 256 weights per block
        down_blocks_per_row = I // 256  # 1024/256 = 4
        down_bytes_per_row = down_blocks_per_row * 176  # 4 * 176 = 704
        down_per_expert = H * down_bytes_per_row  # 4096 * 704 = 2,883,584
        print(f"  down_per_expert={down_per_expert} (C says 2883584)")

        with open(path_dn, 'rb') as f:
            f.seek(ds_dn + info_dn['offset'] + EXPERT_ID * down_per_expert)
            down_data = f.read(down_per_expert)

        # Just compute first 4 rows of down projection for comparison
        expert_out = np.zeros(4, dtype=np.float64)
        for row in range(4):
            row_offset = row * down_bytes_per_row
            row_sum = 0.0
            for b in range(down_blocks_per_row):
                # Q5_K dequant (simplified)
                blk_off = row_offset + b * 176
                d = np.frombuffer(down_data[blk_off:blk_off+2], dtype=np.float16)[0].astype(np.float64)
                dmin = np.frombuffer(down_data[blk_off+2:blk_off+4], dtype=np.float16)[0].astype(np.float64)
                sc_bytes = down_data[blk_off+4:blk_off+16]
                qh = down_data[blk_off+16:blk_off+48]
                qs = down_data[blk_off+48:blk_off+176]

                scales = np.zeros(8, dtype=np.int32)
                mins = np.zeros(8, dtype=np.int32)
                scales[0] = sc_bytes[0] & 63; mins[0] = sc_bytes[4] & 63
                scales[1] = sc_bytes[1] & 63; mins[1] = sc_bytes[5] & 63
                scales[2] = sc_bytes[2] & 63; mins[2] = sc_bytes[6] & 63
                scales[3] = sc_bytes[3] & 63; mins[3] = sc_bytes[7] & 63
                scales[4] = (sc_bytes[8] & 0xF) | ((sc_bytes[0] >> 6) << 4)
                scales[5] = (sc_bytes[9] & 0xF) | ((sc_bytes[1] >> 6) << 4)
                scales[6] = (sc_bytes[10] & 0xF) | ((sc_bytes[2] >> 6) << 4)
                scales[7] = (sc_bytes[11] & 0xF) | ((sc_bytes[3] >> 6) << 4)
                mins[4] = (sc_bytes[8] >> 4) | ((sc_bytes[4] >> 6) << 4)
                mins[5] = (sc_bytes[9] >> 4) | ((sc_bytes[5] >> 6) << 4)
                mins[6] = (sc_bytes[10] >> 4) | ((sc_bytes[6] >> 6) << 4)
                mins[7] = (sc_bytes[11] >> 4) | ((sc_bytes[7] >> 6) << 4)

                for sub in range(8):
                    sc_f = d * float(scales[sub])
                    mn_f = dmin * float(mins[sub])
                    for j in range(16):
                        packed = qs[sub * 16 + j]
                        q0 = (packed & 0x0F) | ((qh[j] >> sub) & 1) << 4
                        q1 = (packed >> 4) | ((qh[j + 16] >> sub) & 1) << 4
                        w0 = sc_f * float(q0) - mn_f
                        w1 = sc_f * float(q1) - mn_f
                        idx0 = b * 256 + sub * 32 + j
                        idx1 = b * 256 + sub * 32 + j + 16
                        if idx0 < I: row_sum += w0 * act[idx0]
                        if idx1 < I: row_sum += w1 * act[idx1]
            expert_out[row] = row_sum

        print(f"\nPython expert_out[0..3] = {expert_out[0]:.6f} {expert_out[1]:.6f} {expert_out[2]:.6f} {expert_out[3]:.6f}")
        print("(Expert 213 has near-zero act, so expert_out should be near-zero too)")

print("\n=== SUMMARY: If all match, layer 0 MoE is correct. Error is in layers 1-59. ===")
