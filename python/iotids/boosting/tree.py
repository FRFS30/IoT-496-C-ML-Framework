"""
iotids.boosting.tree
====================
A single regression tree fit on XGBoost-style second-order gradients.

Mathematical foundation
-----------------------
XGBoost fits each tree by minimizing the regularized objective:

    Obj = sum_j [ -G_j^2 / (2*(H_j + lambda)) ] + gamma * T

where for leaf j:
  G_j = sum of first-order gradients  (g_i = pred_i - y_i  for log-loss)
  H_j = sum of second-order hessians  (h_i = pred_i * (1 - pred_i))
  lambda = L2 regularization on leaf weights
  gamma  = minimum gain required to make a split (pruning threshold)
  T      = number of leaves

The optimal leaf weight for leaf j is:

    w_j* = -G_j / (H_j + lambda)

The gain from splitting node j into left child L and right child R is:

    Gain = 0.5 * [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda)
                   - G^2/(H+lambda) ] - gamma

A split is accepted only if Gain > 0.

Tree structure
--------------
Nodes are stored in a flat list using the implicit binary-heap layout:
  node[0]          = root
  node[2*i + 1]    = left child of node[i]
  node[2*i + 2]    = right child of node[i]

This avoids pointer chasing during inference (important for the server-side
federated simulation), keeps the node array cache-friendly, and makes
leaf-value extraction for FedAvg a simple list comprehension.

The maximum number of nodes in a complete binary tree of depth d is
2^(d+1) - 1.  Nodes that are never reached (because their parent became a
leaf) are stored as make_empty() sentinels.

Column and row subsampling
--------------------------
Both are applied before fit() is called (by GradientBooster), so this class
receives already-subsampled X and residual arrays.  The feature column indices
passed in the sample are in the original (full) feature space; tree.py stores
original indices in SplitNodes so inference on full-width X works correctly.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .node import Node


# ---------------------------------------------------------------------------
# BoostingTree
# ---------------------------------------------------------------------------

class BoostingTree:
    """
    A single gradient-boosted regression tree.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree.  Controls model complexity.
        XGBoost default is 6; typical IDS range is 4–8.
    min_child_weight : float
        Minimum sum of hessians required in a child node before a split
        is accepted.  Acts as a natural sample-count floor (since h_i ≈ 0.25
        for balanced classes, min_child_weight=1 ≈ 4 samples minimum).
    reg_lambda : float
        L2 regularization term added to H_j in the leaf weight formula and
        gain calculation.  Prevents leaf weights from growing unbounded.
    min_gain : float
        Alias for XGBoost's `gamma`.  Splits with gain below this threshold
        are rejected.  Set to 0.0 to disable (pure information-gain splits).
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        min_gain: float = 0.0,
    ) -> None:
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if reg_lambda < 0:
            raise ValueError(f"reg_lambda must be >= 0, got {reg_lambda}")

        self.max_depth: int = max_depth
        self.min_child_weight: float = float(min_child_weight)
        self.reg_lambda: float = float(reg_lambda)
        self.min_gain: float = float(min_gain)

        # Flat node array; populated by fit()
        self._nodes: List[Node] = []
        self._n_features: int = 0   # original (full) feature count
        self._is_fit: bool = False

    # ------------------------------------------------------------------
    # Internal helpers — gain and weight
    # ------------------------------------------------------------------

    def _leaf_weight(self, G: float, H: float) -> float:
        """Optimal leaf weight for leaf with gradient sum G, hessian sum H."""
        return -G / (H + self.reg_lambda)

    def _split_gain(
        self,
        G: float, H: float,
        G_L: float, H_L: float,
        G_R: float, H_R: float,
    ) -> float:
        """
        XGBoost exact gain formula.

        Gain = 0.5 * [ G_L^2/(H_L+λ) + G_R^2/(H_R+λ) - G^2/(H+λ) ] - γ
        """
        score_L = (G_L * G_L) / (H_L + self.reg_lambda) if H_L + self.reg_lambda > 0 else 0.0
        score_R = (G_R * G_R) / (H_R + self.reg_lambda) if H_R + self.reg_lambda > 0 else 0.0
        score   = (G  * G)   / (H  + self.reg_lambda) if H  + self.reg_lambda > 0 else 0.0
        return 0.5 * (score_L + score_R - score) - self.min_gain

    # ------------------------------------------------------------------
    # Internal helpers — best split search
    # ------------------------------------------------------------------

    def _best_split(
        self,
        X: List[List[float]],
        gradients: List[float],
        hessians: List[float],
        indices: List[int],
        feature_cols: List[int],
    ) -> Tuple[int, float, float, List[int], List[int]]:
        """
        Scan all candidate features and thresholds for the best split.

        Returns
        -------
        best_feat  : int   – feature column index in original feature space
                             (-1 if no valid split found)
        best_thresh: float – threshold value
        best_gain  : float – gain achieved
        left_idx   : List[int] – sample indices going left
        right_idx  : List[int] – sample indices going right
        """
        n = len(indices)
        if n == 0:
            return -1, 0.0, -math.inf, [], []

        # Pre-compute node totals
        G_total = sum(gradients[i] for i in indices)
        H_total = sum(hessians[i]  for i in indices)

        best_feat   = -1
        best_thresh = 0.0
        best_gain   = -math.inf
        best_left:  List[int] = []
        best_right: List[int] = []

        for feat in feature_cols:
            # Sort samples by feature value — O(n log n) per feature
            sorted_idx = sorted(indices, key=lambda i: X[i][feat])

            G_L = 0.0
            H_L = 0.0

            for k in range(n - 1):
                i = sorted_idx[k]
                G_L += gradients[i]
                H_L += hessians[i]
                G_R  = G_total - G_L
                H_R  = H_total - H_L

                # Skip if either child has insufficient hessian mass
                if H_L < self.min_child_weight or H_R < self.min_child_weight:
                    continue

                # Skip duplicate thresholds (use midpoint between consecutive values)
                val_k   = X[sorted_idx[k]][feat]
                val_k1  = X[sorted_idx[k + 1]][feat]
                if val_k == val_k1:
                    continue

                threshold = (val_k + val_k1) * 0.5
                gain = self._split_gain(G_total, H_total, G_L, H_L, G_R, H_R)

                if gain > best_gain:
                    best_gain   = gain
                    best_feat   = feat
                    best_thresh = threshold
                    best_left   = sorted_idx[: k + 1]
                    best_right  = sorted_idx[k + 1 :]

        return best_feat, best_thresh, best_gain, best_left, best_right

    # ------------------------------------------------------------------
    # Recursive tree builder
    # ------------------------------------------------------------------

    def _build(
        self,
        X: List[List[float]],
        gradients: List[float],
        hessians: List[float],
        indices: List[int],
        feature_cols: List[int],
        node_idx: int,
        depth: int,
    ) -> None:
        """
        Recursively build the tree into self._nodes[node_idx].

        Ensures self._nodes is large enough before writing, expanding with
        empty sentinels as needed (implicit binary-heap node layout).
        """
        # Expand node list to accommodate this index
        while len(self._nodes) <= node_idx:
            self._nodes.append(Node.make_empty())

        G = sum(gradients[i] for i in indices)
        H = sum(hessians[i]  for i in indices)
        n = len(indices)

        # Leaf conditions: max depth reached, too few samples, or trivial node
        if depth >= self.max_depth or n == 0:
            lv = self._leaf_weight(G, H)
            self._nodes[node_idx] = Node.make_leaf(lv, sample_count=n, sum_hessian=H)
            return

        best_feat, best_thresh, best_gain, left_idx, right_idx = self._best_split(
            X, gradients, hessians, indices, feature_cols
        )

        # No valid split found (gain <= 0 or all thresholds duplicated)
        if best_feat == -1 or best_gain <= 0.0:
            lv = self._leaf_weight(G, H)
            self._nodes[node_idx] = Node.make_leaf(lv, sample_count=n, sum_hessian=H)
            return

        self._nodes[node_idx] = Node.make_split(
            feature_idx=best_feat,
            threshold=best_thresh,
            gain=best_gain,
            sample_count=n,
            sum_hessian=H,
        )

        left_child  = 2 * node_idx + 1
        right_child = 2 * node_idx + 2

        self._build(X, gradients, hessians, left_idx,  feature_cols, left_child,  depth + 1)
        self._build(X, gradients, hessians, right_idx, feature_cols, right_child, depth + 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: List[List[float]],
        gradients: List[float],
        hessians: List[float],
        feature_cols: Optional[List[int]] = None,
    ) -> "BoostingTree":
        """
        Fit this tree on gradient/hessian residuals.

        Parameters
        ----------
        X          : List[List[float]]  – (n_samples, n_features) row-major matrix
        gradients  : List[float]        – first-order gradient per sample (g_i)
        hessians   : List[float]        – second-order hessian per sample (h_i)
        feature_cols : List[int] | None – column indices to consider for splits
                       (column subsampling is applied by GradientBooster before
                       calling fit; pass None to use all columns)

        Returns self for chaining.
        """
        n_samples = len(X)
        if n_samples == 0:
            raise ValueError("fit() received empty dataset")
        if len(gradients) != n_samples or len(hessians) != n_samples:
            raise ValueError(
                f"Length mismatch: X has {n_samples} rows but gradients={len(gradients)}, "
                f"hessians={len(hessians)}"
            )

        self._n_features = len(X[0]) if X else 0
        if feature_cols is None:
            feature_cols = list(range(self._n_features))

        self._nodes = []
        all_indices = list(range(n_samples))
        self._build(X, gradients, hessians, all_indices, feature_cols, node_idx=0, depth=0)
        self._is_fit = True
        return self

    def predict_one(self, x: List[float]) -> float:
        """
        Predict the raw score delta for a single sample.

        Traverses the tree from root to leaf following split decisions.
        Returns the leaf_value accumulated by this tree.
        """
        if not self._is_fit:
            raise RuntimeError("Tree has not been fit yet")

        node_idx = 0
        while node_idx < len(self._nodes):
            node = self._nodes[node_idx]
            if node.is_leaf:
                return node.leaf_value
            # Route sample left or right
            if x[node.feature_idx] <= node.threshold:
                node_idx = 2 * node_idx + 1
            else:
                node_idx = 2 * node_idx + 2

        # Should never reach here on a well-formed tree
        return 0.0

    def predict(self, X: List[List[float]]) -> List[float]:
        """Predict raw score deltas for all samples."""
        return [self.predict_one(x) for x in X]

    # ------------------------------------------------------------------
    # FedAvg interface
    # ------------------------------------------------------------------

    def get_leaves(self) -> List[float]:
        """
        Return a flat list of all leaf values in node-index order.

        This is the unit of federated averaging: all clients in a round
        build trees with the same structure (enforced by round_seed), so
        their leaf arrays are element-wise aligned and can be averaged
        by FedAvgServer.aggregate().
        """
        if not self._is_fit:
            raise RuntimeError("Tree has not been fit yet")
        return [node.leaf_value for node in self._nodes if node.is_leaf]

    def set_leaves(self, values: List[float]) -> None:
        """
        Replace leaf values with the federated-averaged values from the server.

        Parameters
        ----------
        values : List[float] – must be the same length as get_leaves() returns
        """
        leaf_nodes = [node for node in self._nodes if node.is_leaf]
        if len(values) != len(leaf_nodes):
            raise ValueError(
                f"set_leaves() got {len(values)} values but tree has {len(leaf_nodes)} leaves"
            )
        vi = 0
        for node in self._nodes:
            if node.is_leaf:
                node.leaf_value = float(values[vi])
                vi += 1

    # ------------------------------------------------------------------
    # Diagnostics / introspection
    # ------------------------------------------------------------------

    def n_leaves(self) -> int:
        return sum(1 for n in self._nodes if n.is_leaf)

    def n_nodes(self) -> int:
        return len(self._nodes)

    def depth(self) -> int:
        """Actual depth of the built tree (may be less than max_depth)."""
        if not self._nodes:
            return 0
        # Depth of node at index i in implicit binary heap = floor(log2(i+1))
        return max(
            int(math.floor(math.log2(i + 1)))
            for i, node in enumerate(self._nodes)
            if node.is_split()
        ) if any(node.is_split() for node in self._nodes) else 0

    def feature_gains(self, n_features: Optional[int] = None) -> List[float]:
        """
        Per-feature total gain accumulated across all split nodes in this tree.
        Used by GradientBooster to compute feature_importances_.
        """
        nf = n_features if n_features is not None else self._n_features
        gains = [0.0] * nf
        for node in self._nodes:
            if node.is_split() and 0 <= node.feature_idx < nf:
                gains[node.feature_idx] += node.gain
        return gains

    def to_dict(self) -> dict:
        return {
            "max_depth":         self.max_depth,
            "min_child_weight":  self.min_child_weight,
            "reg_lambda":        self.reg_lambda,
            "min_gain":          self.min_gain,
            "n_features":        self._n_features,
            "is_fit":            self._is_fit,
            "nodes":             [n.to_dict() for n in self._nodes],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BoostingTree":
        tree = cls(
            max_depth=d["max_depth"],
            min_child_weight=d["min_child_weight"],
            reg_lambda=d["reg_lambda"],
            min_gain=d["min_gain"],
        )
        tree._n_features = d["n_features"]
        tree._is_fit = d["is_fit"]
        tree._nodes = [Node.from_dict(nd) for nd in d["nodes"]]
        return tree

    def __repr__(self) -> str:
        if not self._is_fit:
            return f"BoostingTree(max_depth={self.max_depth}, [not fit])"
        return (
            f"BoostingTree(max_depth={self.max_depth}, "
            f"depth={self.depth()}, "
            f"n_nodes={self.n_nodes()}, "
            f"n_leaves={self.n_leaves()})"
        )
