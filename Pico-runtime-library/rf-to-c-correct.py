
#takes in the bin file of the file and scaler and produces the C headers for both.  I ended up rolling these two files into just rf_model.h which is why rf_scaler.h isn't in the microcontroller runtime
import struct
from MLlib.python.iotids.forest.serializer import load_rf
from MLlib.python.iotids.utils.io import load

MODEL_PATH = "iotids_rf_model.bin"
SCALER_PATH = "iotids_rf_scaler.bin"

OUTPUT_MODEL_H = "rf_model.h"
OUTPUT_SCALER_H = "rf_scaler.h"


# Load model + scaler
model = load_rf(MODEL_PATH)
scaler_params = load(SCALER_PATH)

medians = scaler_params["medians"]
iqr = scaler_params["iqrs"]

N_FEATURES = len(medians)
N_TREES = len(model.estimators_)


#flatten tree
class CNode:
    def __init__(self, feature, threshold, left, right, value):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def flatten_tree(node):
    nodes = []

    def recurse(n):
        idx = len(nodes)
        nodes.append(None) 

        # Leaf node
        if n.feature is None:
            nodes[idx] = CNode(
                feature=-1,
                threshold=0.0,
                left=-1,
                right=-1,
                value=int(n.value)
            )
            return idx

        # Internal node
        left_idx = recurse(n.left)
        right_idx = recurse(n.right)

        nodes[idx] = CNode(
            feature=int(n.feature),
            threshold=float(n.threshold),
            left=left_idx,
            right=right_idx,
            value=0  # unused
        )

        return idx

    recurse(node)
    return nodes


#export the model
def export_model(model):
    all_trees = []

    for tree in model.estimators_:
        nodes = flatten_tree(tree.root)
        all_trees.append(nodes)

    with open(OUTPUT_MODEL_H, "w") as f:
        f.write("#ifndef RF_MODEL_H\n#define RF_MODEL_H\n\n")

        f.write(f"#define N_TREES {len(all_trees)}\n")
        f.write(f"#define N_FEATURES {N_FEATURES}\n\n")

        f.write("typedef struct {\n")
        f.write("    int feature;\n")
        f.write("    float threshold;\n")
        f.write("    int left;\n")
        f.write("    int right;\n")
        f.write("    int value;\n")
        f.write("} Node;\n\n")

        # Trees
        for t, nodes in enumerate(all_trees):
            f.write(f"static const Node tree_{t}[] = {{\n")
            for n in nodes:
                f.write(
                    f"    {{{n.feature}, {n.threshold:.6f}f, {n.left}, {n.right}, {n.value}}},\n"
                )
            f.write("};\n\n")

        # Forest array
        f.write("static Node* forest[] = {\n")
        for t in range(len(all_trees)):
            f.write(f"    (Node*)tree_{t},\n")
        f.write("};\n\n")

        f.write("#endif\n")

    print(f"✔ Model exported to {OUTPUT_MODEL_H}")


#export the scaler
def export_scaler(medians, iqr):
    with open(OUTPUT_SCALER_H, "w") as f:
        f.write("#ifndef RF_SCALER_H\n#define RF_SCALER_H\n\n")

        f.write(f"#define N_FEATURES {len(medians)}\n\n")

        f.write("static const float medians[N_FEATURES] = {\n    ")
        f.write(", ".join(f"{m:.6f}f" for m in medians))
        f.write("\n};\n\n")

        f.write("static const float iqr[N_FEATURES] = {\n    ")
        f.write(", ".join(f"{s:.6f}f" for s in iqr))
        f.write("\n};\n\n")

        # Faster inference
        f.write("static const float inv_iqr[N_FEATURES] = {\n    ")
        f.write(", ".join(f"{(1.0/s if s != 0 else 0.0):.6f}f" for s in iqr))
        f.write("\n};\n\n")

        f.write("#endif\n")

    print(f"✔ Scaler exported to {OUTPUT_SCALER_H}")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    export_model(model)
    export_scaler(medians, iqr)