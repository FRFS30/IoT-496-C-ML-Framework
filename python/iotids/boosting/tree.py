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

Parallelism
-----------
The dominant cost is the feature scan inside _best_split:

    for each feature:           <- embarrassingly parallel
        sort samples by value   <- O(n log n)
        scan thresholds         <- O(n)

Each feature's scan is completely independent — no shared mutable state,
no communication between features.  This maps directly to a process pool.

A multiprocessing.Pool is created once per BoostingTree.fit() call and
reused across all nodes in that tree (avoiding repeated fork overhead).
On Linux, 'fork' start method means the large X/gradient/hessian arrays
are shared copy-on-write across workers at near-zero memory cost.

n_jobs parameter
----------------
  n_jobs = -1  : use all available cores (default)
  n_jobs =  N  : use exactly N cores
  n_jobs =  1  : disable multiprocessing (serial, for debugging)
"""

from __future__ import annotations

import math
import multiprocessing
import os
from typing import List, Optional, Tuple

from .node import Node


# ---------------------------------------------------------------------------
# Module-level worker — must be top-level (not a method/lambda) for pickle
# ---------------------------------------------------------------------------

def _scan_feature(args):
    """
    Find the best split threshold for a single feature.

    Runs in a worker process. All args are plain Python objects.

    Returns
    -------
    (gain, feat, threshold, left_count)
      gain       : best gain found (-inf if no valid split)
      feat       : feature index (passed through for identification)
      threshold  : best midpoint threshold
      left_count : k+1 where sorted_idx[:k+1] = left child samples
    """
    (feat, sorted_idx, feat_vals, gradients, hessians,
     G_total, H_total, reg_lambda, min_child_weight, min_gain) = args

    n            = len(sorted_idx)
    best_gain    = -math.inf
    best_thresh  = 0.0
    best_left_k  = -1

    G_L = 0.0
    H_L = 0.0

    for k in range(n - 1):
        i    = sorted_idx[k]
        G_L += gradients[i]
        H_L += hessians[i]
        G_R  = G_total - G_L
        H_R  = H_total - H_L

        if H_L < min_child_weight or H_R < min_child_weight:
            continue

        val_k  = feat_vals[k]
        val_k1 = feat_vals[k + 1]
        if val_k == val_k1:
            continue

        # XGBoost exact gain formula
        denom   = H_total + reg_lambda
        denom_L = H_L     + reg_lambda
        denom_R = H_R     + reg_lambda
        score   = (G_total * G_total) / denom   if denom   > 0 else 0.0
        score_L = (G_L     * G_L)     / denom_L if denom_L > 0 else 0.0
        score_R = (G_R     * G_R)     / denom_R if denom_R > 0 else 0.0
        gain    = 0.5 * (score_L + score_R - score) - min_gain

        if gain > best_gain:
            best_gain   = gain
            best_thresh = (val_k + val_k1) * 0.5
            best_left_k = k

    return best_gain, feat, best_thresh, best_left_k


# ---------------------------------------------------------------------------
# BoostingTree
# ---------------------------------------------------------------------------

class BoostingTree:
    """
    A single gradient-boosted regression tree with parallel feature scan.

    Parameters
    ----------
    max_depth        : int   – maximum tree depth
    min_child_weight : float – minimum hessian sum in a child node
    reg_lambda       : float – L2 regularization on leaf weights
    min_gain         : float – minimum gain to accept a split (gamma)
    n_jobs           : int   – worker processes for feature scan
                               -1 = all cores, 1 = serial
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        min_gain: float = 0.0,
        n_jobs: int = -1,
    ) -> None:
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if reg_lambda < 0:
            raise ValueError(f"reg_lambda must be >= 0, got {reg_lambda}")

        self.max_depth        = max_depth
        self.min_child_weight = float(min_child_weight)
        self.reg_lambda       = float(reg_lambda)
        self.min_gain         = float(min_gain)
        self.n_jobs           = n_jobs

        self._nodes:      List[Node] = []
        self._n_features: int        = 0
        self._is_fit:     bool       = False
        self._pool:       Optional[multiprocessing.pool.Pool] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_workers(n_jobs: int, n_features: int) -> int:
        """Cap workers at n_features — never more workers than tasks."""
        if n_jobs == 1:
            return 1
        cpu_count = os.cpu_count() or 1
        workers   = cpu_count if n_jobs == -1 else max(1, n_jobs)
        return min(workers, max(1, n_features))

    def _leaf_weight(self, G: float, H: float) -> float:
        return -G / (H + self.reg_lambda)

    # ------------------------------------------------------------------
    # Best split — parallel across features
    # ------------------------------------------------------------------

    def _best_split(
        self,
        X: List[List[float]],
        gradients: List[float],
        hessians: List[float],
        indices: List[int],
        feature_cols: List[int],
    ) -> Tuple[int, float, float, List[int], List[int]]:
        n = len(indices)
        if n == 0:
            return -1, 0.0, -math.inf, [], []

        G_total = sum(gradients[i] for i in indices)
        H_total = sum(hessians[i]  for i in indices)

        # Pre-sort indices by each feature and extract values.
        # Sorting is done on the main process once per node.
        # The sorted arrays are then shipped to workers (or used serially).
        tasks            = []
        sorted_by_feat   = {}

        for feat in feature_cols:
            s_idx  = sorted(indices, key=lambda i, f=feat: X[i][f])
            f_vals = [X[i][feat] for i in s_idx]
            sorted_by_feat[feat] = s_idx
            tasks.append((
                feat, s_idx, f_vals,
                gradients, hessians,
                G_total, H_total,
                self.reg_lambda, self.min_child_weight, self.min_gain,
            ))

        # Dispatch
        if self._pool is not None and len(tasks) > 1:
            results = self._pool.map(_scan_feature, tasks)
        else:
            results = [_scan_feature(t) for t in tasks]

        # Pick the winning feature
        best_gain   = -math.inf
        best_feat   = -1
        best_thresh = 0.0
        best_left_k = -1

        for gain, feat, thresh, left_k in results:
            if gain > best_gain:
                best_gain   = gain
                best_feat   = feat
                best_thresh = thresh
                best_left_k = left_k

        if best_feat == -1 or best_gain <= 0.0 or best_left_k < 0:
            return -1, 0.0, -math.inf, [], []

        s         = sorted_by_feat[best_feat]
        left_idx  = s[: best_left_k + 1]
        right_idx = s[best_left_k + 1 :]

        return best_feat, best_thresh, best_gain, left_idx, right_idx

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
        while len(self._nodes) <= node_idx:
            self._nodes.append(Node.make_empty())

        G = sum(gradients[i] for i in indices)
        H = sum(hessians[i]  for i in indices)
        n = len(indices)

        if depth >= self.max_depth or n == 0:
            self._nodes[node_idx] = Node.make_leaf(
                self._leaf_weight(G, H), sample_count=n, sum_hessian=H
            )
            return

        best_feat, best_thresh, best_gain, left_idx, right_idx = self._best_split(
            X, gradients, hessians, indices, feature_cols
        )

        if best_feat == -1 or best_gain <= 0.0:
            self._nodes[node_idx] = Node.make_leaf(
                self._leaf_weight(G, H), sample_count=n, sum_hessian=H
            )
            return

        self._nodes[node_idx] = Node.make_split(
            feature_idx=best_feat,
            threshold=best_thresh,
            gain=best_gain,
            sample_count=n,
            sum_hessian=H,
        )

        self._build(X, gradients, hessians, left_idx,  feature_cols, 2*node_idx+1, depth+1)
        self._build(X, gradients, hessians, right_idx, feature_cols, 2*node_idx+2, depth+1)

    # ------------------------------------------------------------------
    # Public: fit
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

        Creates a process pool once, reuses it across all node builds in
        this tree, then closes it cleanly — even if an exception occurs.
        """
        n_samples = len(X)
        if n_samples == 0:
            raise ValueError("fit() received empty dataset")
        if len(gradients) != n_samples or len(hessians) != n_samples:
            raise ValueError(
                f"Length mismatch: X={n_samples}, "
                f"gradients={len(gradients)}, hessians={len(hessians)}"
            )

        self._n_features = len(X[0]) if X else 0
        if feature_cols is None:
            feature_cols = list(range(self._n_features))

        n_workers = self._resolve_workers(self.n_jobs, len(feature_cols))

        self._nodes = []
        try:
            if n_workers > 1:
                # 'fork' is Linux default — workers inherit parent memory
                # (X, gradients, hessians) via copy-on-write at near-zero cost.
                ctx = multiprocessing.get_context("fork")
                self._pool = ctx.Pool(processes=n_workers)
            else:
                self._pool = None

            self._build(
                X, gradients, hessians,
                list(range(n_samples)),
                feature_cols,
                node_idx=0,
                depth=0,
            )
        finally:
            if self._pool is not None:
                self._pool.close()
                self._pool.join()
                self._pool = None

        self._is_fit = True
        return self

    # ------------------------------------------------------------------
    # Public: predict
    # ------------------------------------------------------------------

    def predict_one(self, x: List[float]) -> float:
        if not self._is_fit:
            raise RuntimeError("Tree has not been fit yet")
        node_idx = 0
        while node_idx < len(self._nodes):
            node = self._nodes[node_idx]
            if node.is_leaf:
                return node.leaf_value
            node_idx = (2*node_idx+1) if x[node.feature_idx] <= node.threshold \
                       else (2*node_idx+2)
        return 0.0

    def predict(self, X: List[List[float]]) -> List[float]:
        return [self.predict_one(x) for x in X]

    # ------------------------------------------------------------------
    # FedAvg interface
    # ------------------------------------------------------------------

    def get_leaves(self) -> List[float]:
        if not self._is_fit:
            raise RuntimeError("Tree has not been fit yet")
        return [node.leaf_value for node in self._nodes if node.is_leaf]

    def set_leaves(self, values: List[float]) -> None:
        leaf_nodes = [node for node in self._nodes if node.is_leaf]
        if len(values) != len(leaf_nodes):
            raise ValueError(
                f"set_leaves() got {len(values)} values but tree has "
                f"{len(leaf_nodes)} leaves"
            )
        vi = 0
        for node in self._nodes:
            if node.is_leaf:
                node.leaf_value = float(values[vi])
                vi += 1

    # ------------------------------------------------------------------
    # Diagnostics / serialization
    # ------------------------------------------------------------------

    def n_leaves(self) -> int:
        return sum(1 for n in self._nodes if n.is_leaf)

    def n_nodes(self) -> int:
        return len(self._nodes)

    def depth(self) -> int:
        if not self._nodes:
            return 0
        return max(
            int(math.floor(math.log2(i + 1)))
            for i, node in enumerate(self._nodes)
            if node.is_split()
        ) if any(node.is_split() for node in self._nodes) else 0

    def feature_gains(self, n_features: Optional[int] = None) -> List[float]:
        nf = n_features if n_features is not None else self._n_features
        gains = [0.0] * nf
        for node in self._nodes:
            if node.is_split() and 0 <= node.feature_idx < nf:
                gains[node.feature_idx] += node.gain
        return gains

    def to_dict(self) -> dict:
        return {
            "max_depth":        self.max_depth,
            "min_child_weight": self.min_child_weight,
            "reg_lambda":       self.reg_lambda,
            "min_gain":         self.min_gain,
            "n_features":       self._n_features,
            "is_fit":           self._is_fit,
            "nodes":            [n.to_dict() for n in self._nodes],
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
        tree._is_fit     = d["is_fit"]
        tree._nodes      = [Node.from_dict(nd) for nd in d["nodes"]]
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