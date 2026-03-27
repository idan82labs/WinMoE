"""
Patch GGUF file to change num_experts_per_tok (K) value.

This modifies the GGUF metadata to set a different K for MoE routing.
llama.cpp reads this value to determine how many experts to activate.

Usage:
  python patch_k_experts.py <gguf_file> --k 5
"""
import argparse
import struct
import sys


def read_gguf_header(f):
    """Read GGUF file header."""
    magic = f.read(4)
    if magic != b'GGUF':
        raise ValueError(f"Not a GGUF file (magic: {magic})")
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    return version, n_tensors, n_kv


def read_string(f):
    """Read a GGUF string (length-prefixed)."""
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')


def read_value(f, vtype):
    """Read a GGUF value by type."""
    type_readers = {
        0: lambda: struct.unpack('<B', f.read(1))[0],    # UINT8
        1: lambda: struct.unpack('<b', f.read(1))[0],    # INT8
        2: lambda: struct.unpack('<H', f.read(2))[0],    # UINT16
        3: lambda: struct.unpack('<h', f.read(2))[0],    # INT16
        4: lambda: struct.unpack('<I', f.read(4))[0],    # UINT32
        5: lambda: struct.unpack('<i', f.read(4))[0],    # INT32
        6: lambda: struct.unpack('<f', f.read(4))[0],    # FLOAT32
        7: lambda: struct.unpack('<?', f.read(1))[0],    # BOOL
        8: lambda: read_string(f),                        # STRING
        10: lambda: struct.unpack('<Q', f.read(8))[0],   # UINT64
        11: lambda: struct.unpack('<q', f.read(8))[0],   # INT64
        12: lambda: struct.unpack('<d', f.read(8))[0],   # FLOAT64
    }
    if vtype == 9:  # ARRAY
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        return [read_value(f, arr_type) for _ in range(arr_len)]
    if vtype in type_readers:
        return type_readers[vtype]()
    raise ValueError(f"Unknown type: {vtype}")


def find_and_patch_k(filepath, new_k, dry_run=True):
    """Find and patch num_experts_per_tok in a GGUF file."""
    target_keys = [
        'qwen3moe.expert_used_count',
        'qwen3_5moe.expert_used_count',
        'llama.expert_used_count',
        'general.expert_used_count',
    ]

    with open(filepath, 'r+b') as f:
        version, n_tensors, n_kv = read_gguf_header(f)
        print(f"GGUF v{version}, {n_tensors} tensors, {n_kv} KV pairs")

        for i in range(n_kv):
            key = read_string(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            value_offset = f.tell()
            value = read_value(f, vtype)

            if 'expert' in key.lower():
                print(f"  [{i}] {key} = {value} (type={vtype}, offset={value_offset})")

            if any(key == tk for tk in target_keys) or key.endswith('.expert_used_count'):
                print(f"\n  FOUND: {key} = {value} at offset {value_offset}")
                if value == new_k:
                    print(f"  Already set to {new_k}, no change needed.")
                    return True

                if dry_run:
                    print(f"  DRY RUN: Would change {value} -> {new_k}")
                else:
                    # Write new value
                    f.seek(value_offset)
                    if vtype == 4:  # UINT32
                        f.write(struct.pack('<I', new_k))
                    elif vtype == 5:  # INT32
                        f.write(struct.pack('<i', new_k))
                    else:
                        print(f"  Cannot patch type {vtype}")
                        return False
                    print(f"  PATCHED: {value} -> {new_k}")
                return True

    print("  Key not found in GGUF metadata!")
    return False


def main():
    parser = argparse.ArgumentParser(description="Patch GGUF K value")
    parser.add_argument("gguf_file", help="Path to first GGUF shard")
    parser.add_argument("--k", type=int, default=5, help="New K value")
    parser.add_argument("--apply", action="store_true", help="Actually apply (default is dry run)")
    args = parser.parse_args()

    print(f"Patching {args.gguf_file}")
    print(f"Target K: {args.k}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print()

    find_and_patch_k(args.gguf_file, args.k, dry_run=not args.apply)


if __name__ == "__main__":
    main()
