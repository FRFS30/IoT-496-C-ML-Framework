import struct
import array


# ------------------------------------------------------------------ #
# Compact binary serialiser for Random Forest
# Avoids pickle — replaces ~5.5 MB pickle with a tighter layout.
# float64 -> float32 downcast on save.
# ------------------------------------------------------------------ #
MAGIC = b"IOTRF\x01"


def save_rf(model, path):
    """Serialise a RandomForestClassifier to a compact binary file."""
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(">I", model.n_estimators))
        for tree in model.estimators_:
            nodes = tree.get_params()
            f.write(struct.pack(">I", len(nodes)))
            for node in nodes:
                if node is None:
                    f.write(b"\x00")
                else:
                    f.write(b"\x01")
                    feat = node["feature"] if node["feature"] is not None else -1
                    f.write(struct.pack(">i", feat))
                    # float32 downcast — saves space vs float64
                    thresh = node["threshold"] if node["threshold"] is not None else 0.0
                    f.write(struct.pack(">f", float(thresh)))
                    f.write(struct.pack(">i", int(node["value"]) if node["value"] is not None else 0))
                    f.write(struct.pack(">f", float(node["prob"]) if node["prob"] is not None else 0.0))


def load_rf(path):
    """Load a RandomForestClassifier from a compact binary file."""
    from .random_forest import RandomForestClassifier
    from .decision_tree import DecisionTree

    with open(path, "rb") as f:
        magic = f.read(len(MAGIC))
        assert magic == MAGIC, "Bad magic — not an iotids RF file"

        n_trees = struct.unpack(">I", f.read(4))[0]
        model = RandomForestClassifier(n_estimators=n_trees)
        model.estimators_ = []

        for _ in range(n_trees):
            n_nodes = struct.unpack(">I", f.read(4))[0]
            nodes = []
            for _ in range(n_nodes):
                is_real = ord(f.read(1))
                if not is_real:
                    nodes.append(None)
                else:
                    feat   = struct.unpack(">i", f.read(4))[0]
                    thresh = struct.unpack(">f", f.read(4))[0]
                    value  = struct.unpack(">i", f.read(4))[0]
                    prob   = struct.unpack(">f", f.read(4))[0]
                    nodes.append({
                        "feature":   feat if feat >= 0 else None,
                        "threshold": thresh if feat >= 0 else None,
                        "value":     value,
                        "prob":      prob,
                    })

            tree = DecisionTree()
            tree.set_params(nodes)
            model.estimators_.append(tree)

    return model
