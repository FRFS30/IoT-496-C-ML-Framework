"""
Xgb_inference.py
Exports a trained XGBoostClassifier to a minimal binary format.
Works with both old heap-indexed models and new compact models.
"""
import math, os, struct, sys, glob, random

def float_to_f16_bits(v):
    if math.isnan(v) or abs(v) > 65504: return 0x7E00
    if math.isinf(v): return 0x7C00 if v > 0 else 0xFC00
    f32_bits = struct.unpack(">I", struct.pack(">f", float(v)))[0]
    sign   = (f32_bits >> 31) & 0x1
    exp32  = (f32_bits >> 23) & 0xFF
    mant32 = f32_bits & 0x7FFFFF
    exp16  = exp32 - 127 + 15
    if exp16 >= 31: return (sign << 15) | 0x7C00
    if exp16 <= 0:  return sign << 15
    return (sign << 15) | (exp16 << 10) | (mant32 >> 13)

def f16_bits_to_float(bits):
    sign  = (bits >> 15) & 0x1
    exp16 = (bits >> 10) & 0x1F
    mant  = bits & 0x3FF
    if exp16 == 0x1F:
        return float("nan") if mant else (float("-inf") if sign else float("inf"))
    val = ((1+mant/1024.0)*(2**(exp16-15))) if exp16 else (mant/1024.0*2**-14)
    return -val if sign else val

def pack_f16(v):
    b = float_to_f16_bits(v)
    return bytes([(b >> 8) & 0xFF, b & 0xFF])

def load_model(path):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))
    from iotids.boosting.serializer import load_xgb
    return load_xgb(path)

def get_reachable_bfs(nodes):
    """BFS traversal -- works for both heap and compact layouts."""
    if not nodes:
        return []
    result = []
    queue = [0]
    visited = set()
    while queue:
        idx = queue.pop(0)
        if idx in visited or idx < 0 or idx >= len(nodes):
            continue
        visited.add(idx)
        n = nodes[idx]
        result.append(idx)
        thresh = getattr(n, "threshold", 0.0)
        feat   = getattr(n, "feature_idx", -1)
        if n.is_leaf or feat < 0 or abs(thresh) > 1e12:
            continue
        # Check for compact (explicit child pointers) vs heap layout
        lc = getattr(n, "left_child",  -1)
        rc = getattr(n, "right_child", -1)
        if lc >= 0 and rc >= 0:
            queue.append(lc)
            queue.append(rc)
        else:
            queue.append(2*idx+1)
            queue.append(2*idx+2)
    return result

MAGIC   = b"IXGB"
VERSION = 3  # v3: explicit left/right child indices per node

def export_inference(model, out_path):
    """
    Format (little-endian):
      Header   : 16 bytes
      Tree idx : n_trees * 4 bytes (byte offsets into node block)
      Nodes    : variable -- only reachable nodes, compact
        Split: [feat u8][thresh f16 hi][thresh f16 lo][left u16 hi][left u16 lo][right u16 hi][right u16 lo] = 7 bytes
        Leaf:  [0xFF][val f16 hi][val f16 lo] = 3 bytes
    """
    booster    = model._booster
    trees      = booster._trees
    n_trees    = len(trees)
    n_features = trees[0]._n_features if trees else 0
    opt_thresh = getattr(model, "_optimal_threshold", 0.5)
    base_score = getattr(booster, "base_score", 0.5)

    tree_blocks = []
    for tree in trees:
        nodes     = tree._nodes
        reachable = get_reachable_bfs(nodes)

        # Map old index -> new compact index
        old_to_new = {old: new for new, old in enumerate(reachable)}

        block = bytearray()
        for old_idx in reachable:
            n     = nodes[old_idx]
            feat  = getattr(n, "feature_idx", -1)
            thresh = getattr(n, "threshold", 0.0)
            if n.is_leaf or feat < 0 or abs(thresh) > 1e12:
                block += b"\xFF" + pack_f16(n.leaf_value if n.is_leaf else 0.0)
                # Pad to 7 bytes to keep fixed-size stride
                block += b"\x00\x00\x00\x00"
            else:
                lc = getattr(n, "left_child",  -1)
                rc = getattr(n, "right_child", -1)
                if lc < 0:  # heap layout -- compute from position
                    lc = 2*old_idx + 1
                    rc = 2*old_idx + 2
                new_lc = old_to_new.get(lc, 0xFFFF)
                new_rc = old_to_new.get(rc, 0xFFFF)
                block += (bytes([feat & 0xFF]) + pack_f16(thresh)
                          + struct.pack(">HH", new_lc & 0xFFFF, new_rc & 0xFFFF))
        tree_blocks.append(bytes(block))

    header = struct.pack("<4sBHBHHxxxx",
        MAGIC, VERSION, n_trees, n_features,
        float_to_f16_bits(opt_thresh),
        float_to_f16_bits(base_score))
    assert len(header) == 16

    index_bytes = bytearray()
    offset = 0
    for block in tree_blocks:
        index_bytes += struct.pack("<I", offset)
        offset += len(block)

    node_block = b"".join(tree_blocks)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(index_bytes)
        f.write(node_block)

    return len(header) + len(index_bytes) + len(node_block)

def verify_export(model, out_path, n_samples=2000):
    with open(out_path, "rb") as f:
        raw = f.read()

    n_trees    = struct.unpack("<H", raw[5:7])[0]
    n_feat     = raw[7]
    opt_thresh = f16_bits_to_float(struct.unpack("<H", raw[8:10])[0])
    base_score = f16_bits_to_float(struct.unpack("<H", raw[10:12])[0])
    assert raw[0:4] == MAGIC

    idx_start  = 16
    node_start = idx_start + n_trees * 4
    offsets    = [struct.unpack("<I", raw[idx_start+i*4:idx_start+i*4+4])[0]
                  for i in range(n_trees)]

    def predict_one(x):
        score = base_score
        for t in range(n_trees):
            base = node_start + offsets[t]
            ni = 0
            while True:
                pos = base + ni * 7
                b0 = raw[pos]
                if b0 == 0xFF:
                    score += f16_bits_to_float((raw[pos+1] << 8) | raw[pos+2])
                    break
                thresh = f16_bits_to_float((raw[pos+1] << 8) | raw[pos+2])
                lc = (raw[pos+3] << 8) | raw[pos+4]
                rc = (raw[pos+5] << 8) | raw[pos+6]
                ni = lc if x[b0] <= thresh else rc
        prob = 1.0 / (1.0 + math.exp(-score))
        return 1 if prob >= opt_thresh else 0

    random.seed(42)
    mismatches = 0
    for _ in range(n_samples):
        x = [random.uniform(-3.0, 3.0) for _ in range(n_feat)]
        orig = model.predict([x])[0]
        exp  = predict_one(x)
        if orig != exp:
            mismatches += 1

    rate = mismatches / n_samples
    print(f"  Verification: {n_samples} samples, {mismatches} flips ({rate*100:.2f}%)")
    print(f"  {'OK' if rate <= 0.01 else 'WARNING >1% flips'}")
    return rate

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("models/xgb_baseline_*.bin"))[-1]
    print(f"Loading: {src}")
    model = load_model(src)

    booster = model._booster
    trees   = booster._trees
    n_trees = len(trees)

    reachable_counts = [len(get_reachable_bfs(t._nodes)) for t in trees]
    total_reach = sum(reachable_counts)

    # v3 format: each node is 7 bytes (split or leaf padded to 7)
    est = 16 + n_trees*4 + total_reach*7
    print(f"  Trees: {n_trees}")
    print(f"  Reachable nodes/tree (avg): {total_reach/n_trees:.1f}")
    print(f"  Estimated output size: {est/1024:.1f} KB")

    out  = "models/xgb_inference.bin"
    size = export_inference(model, out)
    print(f"  Exported: {out}  ({size/1024:.1f} KB)")

    print("Verifying...")
    verify_export(model, out)