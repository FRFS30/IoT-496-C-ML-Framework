import sys
sys.path.insert(0, "python")
from iotids.boosting.serializer import load_xgb
from collections import Counter

model = load_xgb("models/xgb_baseline_s010_20260316_224814.bin")
tree = model._booster._trees[0]

print(f"Tree 0 has {len(tree._nodes)} nodes")
print()
print("First 10 nodes:")
for i, n in enumerate(tree._nodes[:10]):
    thresh = round(n.threshold, 4) if hasattr(n, "threshold") else "N/A"
    print(f"  [{i}] is_leaf={n.is_leaf} feature_idx={getattr(n,'feature_idx',-99)} "
          f"threshold={thresh} leaf_value={round(n.leaf_value, 4)}")

print()
print("Node fields:", vars(tree._nodes[0]))
print()

# Classify all nodes across all trees
all_kinds = Counter()
for t in model._booster._trees:
    for n in t._nodes:
        fid = getattr(n, "feature_idx", -99)
        if n.is_leaf and n.leaf_value == 0.0 and fid == -1:
            all_kinds["empty"] += 1
        elif n.is_leaf:
            all_kinds["real_leaf"] += 1
        else:
            all_kinds["split"] += 1

print("Node type counts across all trees:")
for k, v in all_kinds.items():
    print(f"  {k}: {v}")

total = sum(all_kinds.values())
real = all_kinds["real_leaf"] + all_kinds["split"]
print(f"\nTotal nodes: {total}")
print(f"Real nodes:  {real}")
print(f"Empty nodes: {all_kinds['empty']}")
print(f"Estimated size with empties stripped: {(16 + 200*4 + real*3)/1024:.1f} KB")