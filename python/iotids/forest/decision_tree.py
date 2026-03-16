import math


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "prob")

    def __init__(self):
        self.feature   = None
        self.threshold = None
        self.left      = None
        self.right     = None
        self.value     = None   # majority class at leaf
        self.prob      = None   # P(class=1) at leaf


class DecisionTree:
    """
    Binary CART tree using Gini impurity.
    Serialisable node structure for FedAvg and compact storage.
    """

    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features  # int or None (use all)
        self.root = None

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        self.root = self._grow(X, y, depth=0)
        return self
    def _grow(self, X, y, depth):
        n = len(y)
        n_pos = sum(y)
        n_neg = n - n_pos

        node = _Node()
        node.prob  = n_pos / n if n > 0 else 0.0
        node.value = 1 if n_pos >= n_neg else 0

        if (n < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)
                or n_pos == 0 or n_neg == 0):
            return node

        n_features = len(X[0])
        if self.max_features is None:
            feat_indices = list(range(n_features))
        else:
            feat_indices = _sample_without_replace(n_features, self.max_features)

        best_feat, best_thresh, best_gain = None, None, -1.0
        parent_gini = _gini(y)

        for f in feat_indices:
            # Sort once by feature value — O(n log n) instead of O(n * thresholds)
            order = sorted(range(n), key=lambda i: X[i][f])
            f_vals = [X[i][f] for i in order]
            f_y    = [y[i]    for i in order]

            # Running counts — O(n) scan instead of rebuilding left/right each time
            left_pos  = 0
            left_n    = 0
            right_pos = n_pos
            right_n   = n

            for j in range(n - 1):
                left_pos  += f_y[j]
                left_n    += 1
                right_pos -= f_y[j]
                right_n   -= 1

                # Skip duplicate feature values (no valid split here)
                if f_vals[j] == f_vals[j + 1]:
                    continue
                # Enforce min_samples_leaf
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue

                left_neg  = left_n  - left_pos
                right_neg = right_n - right_pos

                left_gini  = 1.0 - (left_pos/left_n)**2   - (left_neg/left_n)**2
                right_gini = 1.0 - (right_pos/right_n)**2 - (right_neg/right_n)**2
                gain = parent_gini - (left_n/n * left_gini + right_n/n * right_gini)

                if gain > best_gain:
                    best_gain   = gain
                    best_feat   = f
                    best_thresh = (f_vals[j] + f_vals[j + 1]) / 2.0

        if best_feat is None:
            return node

        left_mask  = [i for i in range(n) if X[i][best_feat] <= best_thresh]
        right_mask = [i for i in range(n) if X[i][best_feat] >  best_thresh]

        node.feature   = best_feat
        node.threshold = best_thresh
        node.left  = self._grow([X[i] for i in left_mask],  [y[i] for i in left_mask],  depth + 1)
        node.right = self._grow([X[i] for i in right_mask], [y[i] for i in right_mask], depth + 1)
        return node
    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #
    def predict(self, X):
        return [self._traverse(row, self.root).value for row in X]

    def predict_proba(self, X):
        return [self._traverse(row, self.root).prob for row in X]

    def _traverse(self, row, node):
        if node.feature is None:
            return node
        if row[node.feature] <= node.threshold:
            return self._traverse(row, node.left)
        return self._traverse(row, node.right)

    # ------------------------------------------------------------------ #
    # Serialisable params (depth-first node list)
    # ------------------------------------------------------------------ #
    def get_params(self):
        nodes = []
        _serialize_node(self.root, nodes)
        return nodes

    def set_params(self, nodes):
        self.root, _ = _deserialize_node(nodes, 0)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _gini(y):
    n = len(y)
    if n == 0:
        return 0.0
    p = sum(y) / n
    return 1.0 - p * p - (1.0 - p) * (1.0 - p)


def _sample_without_replace(n, k):
    import random
    pool = list(range(n))
    result = []
    for _ in range(min(k, n)):
        i = random.randint(0, len(pool) - 1)
        result.append(pool.pop(i))
    return result


def _serialize_node(node, out):
    if node is None:
        out.append(None)
        return
    out.append({
        "feature":   node.feature,
        "threshold": node.threshold,
        "value":     node.value,
        "prob":      node.prob,
    })
    _serialize_node(node.left,  out)
    _serialize_node(node.right, out)


def _deserialize_node(nodes, idx):
    if idx >= len(nodes) or nodes[idx] is None:
        return None, idx + 1
    d = nodes[idx]
    node = _Node()
    node.feature   = d["feature"]
    node.threshold = d["threshold"]
    node.value     = d["value"]
    node.prob      = d["prob"]
    node.left,  idx = _deserialize_node(nodes, idx + 1)
    node.right, idx = _deserialize_node(nodes, idx)
    return node, idx
