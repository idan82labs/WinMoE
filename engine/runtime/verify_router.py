"""Verify MoE router expert selection for layer 0, token 0."""
import numpy as np
import struct, os

GGUF_DIR = "D:/models/qwen35-397b-q4/Q4_K_M"
GGUF_BASE = "Qwen3.5-397B-A17B-Q4_K_M"

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
            # Skip KV
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
            # Parse tensors
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

# Read MoE normed input from binary dump
normed = np.fromfile("moe_normed_L0.bin", dtype=np.float32)
print(f"MoE normed: rms={np.sqrt(np.mean(normed**2)):.4f} [0..3]={normed[0]:.4f} {normed[1]:.4f} {normed[2]:.4f} {normed[3]:.4f}")

# Read router weights for layer 0 (FP32)
path, ds, info = find_tensor("blk.0.ffn_gate_inp.weight")
if info:
    print(f"Router: dims={info['dims']} type={info['type']} in {os.path.basename(path)}")
    H = 4096
    N_EXPERTS = 512
    with open(path, 'rb') as f:
        f.seek(ds + info['offset'])
        if info['type'] == 0:  # F32
            router_w = np.frombuffer(f.read(N_EXPERTS * H * 4), dtype=np.float32).reshape(N_EXPERTS, H)
        else:
            print(f"  Router type {info['type']} not F32, skipping")
            exit()

    # Compute router logits
    logits = router_w @ normed.astype(np.float64)

    # Top-10
    top10 = np.argsort(logits)[-10:][::-1]
    print(f"\nPython top-10 experts:")
    for i, eid in enumerate(top10):
        print(f"  #{i+1}: expert {eid} logit={logits[eid]:.4f}")

    # Softmax on top-10
    top_logits = logits[top10]
    top_exp = np.exp(top_logits - top_logits.max())
    top_weights = top_exp / top_exp.sum()
    print(f"\nSoftmax weights: {top_weights[:3]}")

    print(f"\nC engine: experts=[213,136,53] weights=[0.1785,0.1648,0.1358]")
    print(f"Match top-3: {top10[0]}=={213}, {top10[1]}=={136}, {top10[2]}=={53}")
else:
    print("Router weight not found!")
