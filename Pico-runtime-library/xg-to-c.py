import math
import struct

# Helper functions for handling half-precision floats
def float_to_f16_bits(v):
    if math.isnan(v) or abs(v) > 65504:
        return 0x7E00
    if math.isinf(v):
        return 0x7C00 if v > 0 else 0xFC00
    f32_bits = struct.unpack(">I", struct.pack(">f", float(v)))[0]
    sign = (f32_bits >> 31) & 0x1
    exp32 = (f32_bits >> 23) & 0xFF
    mant32 = f32_bits & 0x7FFFFF
    exp16 = exp32 - 127 + 15
    if exp16 >= 31:
        return (sign << 15) | 0x7C00
    if exp16 <= 0:
        return sign << 15
    return (sign << 15) | (exp16 << 10) | (mant32 >> 13)

def pack_f16(v):
    b = float_to_f16_bits(v)
    return bytes([(b >> 8) & 0xFF, b & 0xFF])

# Define the structure for nodes
class Node:
    def __init__(self, feature, threshold, left, right, value):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Leaf value or 0 for split nodes

# Function to serialize a tree into C header code
def export_tree_to_c(trees, output_file="xgb_model.h"):
    # Open the C header file for writing
    with open(output_file, "w") as f:
        f.write("#ifndef XGB_MODEL_H\n#define XGB_MODEL_H\n\n")
        
        # Define the structure of a node in the tree
        f.write("typedef struct {\n")
        f.write("    int feature;\n")
        f.write("    float threshold;\n")  # Store threshold as float (use half precision for inference)
        f.write("    int left;\n")
        f.write("    int right;\n")
        f.write("    float value;\n")  # Leaf value for leaf nodes
        f.write("} Node;\n\n")

        # Write the number of trees and features
        n_trees = len(trees)
        n_features = len(trees[0]['nodes'])  # Assuming all trees have the same number of features
        f.write(f"#define N_TREES {n_trees}\n")
        f.write(f"#define N_FEATURES {n_features}\n\n")

        # Serialize each tree
        for tree_idx, tree in enumerate(trees):
            f.write(f"static const Node tree_{tree_idx}[] = {{\n")
            for node in tree['nodes']:
                # For split nodes
                if node['is_leaf']:
                    # Leaf node
                    leaf_value = node['value']
                    f.write(f"    {{-1, {leaf_value:.6f}f, -1, -1, {leaf_value:.6f}f}},\n")
                else:
                    # Split node
                    feature = node['feature']
                    threshold = node['threshold']
                    left = node['left']
                    right = node['right']
                    f.write(f"    {{ {feature}, {threshold:.6f}f, {left}, {right}, 0.0f }},\n")
            f.write("};\n\n")

        # Forest array (array of pointers to the trees)
        f.write("static Node* forest[] = {\n")
        for t in range(n_trees):
            f.write(f"    (Node*)tree_{t},\n")
        f.write("};\n\n")

        f.write("#endif\n")

    print(f"Model exported to {output_file}")

# Example usage with mock data
def example_usage():
    trees = [
        # Tree 0
        {
            'nodes': [
                {'is_leaf': False, 'feature': 0, 'threshold': 0.5, 'left': 1, 'right': 2, 'value': 0.0},
                {'is_leaf': True, 'feature': None, 'threshold': None, 'left': None, 'right': None, 'value': 1.0},
                {'is_leaf': True, 'feature': None, 'threshold': None, 'left': None, 'right': None, 'value': -1.0}
            ]
        },
        # Tree 1
        {
            'nodes': [
                {'is_leaf': False, 'feature': 1, 'threshold': 1.5, 'left': 3, 'right': 4, 'value': 0.0},
                {'is_leaf': True, 'feature': None, 'threshold': None, 'left': None, 'right': None, 'value': 2.0},
                {'is_leaf': True, 'feature': None, 'threshold': None, 'left': None, 'right': None, 'value': -2.0}
            ]
        }
    ]
    
    export_tree_to_c(trees)

if __name__ == "__main__":
    example_usage()