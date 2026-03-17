"""
iotids.boosting.tree
====================
A single regression tree fit on XGBoost-style second-order gradients.

Speedup strategy: pre-sorted column indices
-------------------------------------------
The previous implementation sorted each feature column at every node:
O(n log n) per feature per node with a Python lambda called n times.
With 222K samples and 24 features this was ~30 seconds per tree.

This implementation pre-sorts ALL feature columns ONCE before building
the tree, then at each node filters the pre-sorted order to active rows
in O(n_active) using a Python set for O(1) membership checks.

No sorting happens inside the workers -- only linear scans.
"""

from __future__ import annotations

import math
import multiprocessing
import os
from typing import FrozenSet, List, Optional, Tuple

from .node import Node


# ---------------------------------------------------------------------------
# Module-level shared state -- populated BEFORE pool is forked
# ---------------------------------------------------------------------------

_SORTED_COLS: List[List[int]]   = []   # [feat] -> row indices sorted by X[:,feat]
_SHARED_G:    List[float]       = []
_SHARED_H:    List[float]       = []
_FEAT_VALS:   List[List[float]] = []   # [feat][row] = value (column-major cache)


# ---------------------------------------------------------------------------
# Worker -- O(n_active) linear scan, no sort
# ---------------------------------------------------------------------------

def _scan_feature_global(args):
    feat, active_set, G_total, H_total, lam, mcw, mg = args

    sorted_all = _SORTED_COLS[feat]
    feat_vals  = _FEAT_VALS[feat]
    g          = _SHARED_G
    h          = _SHARED_H

    # Filter global pre-sorted order to active rows -- O(n_active)
    sorted_idx = [i for i in sorted_all if i in active_set]
    n = len(sorted_idx)

    if n < 2:
        return -math.inf, feat, 0.0, -1

    best_gain   = -math.inf
    best_thresh = 0.0
    best_left_k = -1
    G_L = 0.0
    H_L = 0.0

    for k in range(n - 1):
        i    = sorted_idx[k]
        G_L += g[i]
        H_L += h[i]
        H_R  = H_total - H_L

        if H_L < mcw or H_R < mcw:
            continue

        v_k  = feat_vals[sorted_idx[k]]
        v_k1 = feat_vals[sorted_idx[k + 1]]
        if v_k == v_k1:
            continue

        G_R  = G_total - G_L
        d    = H_total + lam
        dL   = H_L     + lam
        dR   = H_R     + lam
        gain = 0.5 * (
            G_L * G_L / dL +
            G_R * G_R / dR -
            G_total * G_total / d
        ) - mg

        if gain > best_gain:
            best_gain   = gain
            best_thresh = (v_k + v_k1) * 0.5
            best_left_k = k

    return best_gain, feat, best_thresh, best_left_k


# ---------------------------------------------------------------------------
# BoostingTree
# ---------------------------------------------------------------------------

class BoostingTree:
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

    @staticmethod
    def _resolve_workers(n_jobs: int, n_features: int) -> int:
        if n_jobs == 1:
            return 1
        cpu_count = os.cpu_count() or 1
        workers   = cpu_count if n_jobs == -1 else max(1, n_jobs)
        return min(workers, max(1, n_features))

    def _leaf_weight(self, G: float, H: float) -> float:
        return -G / (H + self.reg_lambda)

    def _best_split(
        self,
        indices: List[int],
        feature_cols: List[int],
        G_total: float,
        H_total: float,
    ) -> Tuple[int, float, float, List[int], List[int]]:
        n = len(indices)
        if n < 2:
            return -1, 0.0, -math.inf, [], []

        active_set = frozenset(indices)
        lam = self.reg_lambda
        mcw = self.min_child_weight
        mg  = self.min_gain

        tasks = [
            (feat, active_set, G_total, H_total, lam, mcw, mg)
            for feat in feature_cols
        ]

        if self._pool is not None:
            results = self._pool.map(_scan_feature_global, tasks)
        else:
            results = [_scan_feature_global(t) for t in tasks]

        best_gain, best_feat, best_thresh, best_left_k = max(
            results, key=lambda r: r[0]
        )

        if best_feat == -1 or best_gain <= 0.0 or best_left_k < 0:
            return -1, 0.0, -math.inf, [], []

        sorted_winner = [i for i in _SORTED_COLS[best_feat] if i in active_set]
        left_idx  = sorted_winner[: best_left_k + 1]
        right_idx = sorted_winner[best_left_k + 1 :]

        return best_feat, best_thresh, best_gain, left_idx, right_idx

    def _build(
        self,
        indices: List[int],
        feature_cols: List[int],
        node_idx: int,
        depth: int,
    ) -> None:
        g = _SHARED_G
        h = _SHARED_H

        while len(self._nodes) <= node_idx:
            self._nodes.append(Node.make_empty())

        G = sum(g[i] for i in indices)
        H = sum(h[i] for i in indices)
        n = len(indices)

        if depth >= self.max_depth or n == 0:
            self._nodes[node_idx] = Node.make_leaf(
                self._leaf_weight(G, H), sample_count=n, sum_hessian=H
            )
            return

        best_feat, best_thresh, best_gain, left_idx, right_idx = self._best_split(
            indices, feature_cols, G, H
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

        self._build(left_idx,  feature_cols, 2*node_idx+1, depth+1)
        self._build(right_idx, feature_cols, 2*node_idx+2, depth+1)

    def fit(
        self,
        X: List[List[float]],
        gradients: List[float],
        hessians: List[float],
        feature_cols: Optional[List[int]] = None,
    ) -> "BoostingTree":
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

        # Pre-sort all feature columns once (replaces per-node O(n log n) sorts)
        global _SORTED_COLS, _SHARED_G, _SHARED_H, _FEAT_VALS

        # Build column-major value cache for fast worker access
        _FEAT_VALS = [[X[i][f] for i in range(n_samples)] for f in range(self._n_features)]
        # Sort each feature column once
        _SORTED_COLS = [
            sorted(range(n_samples), key=lambda i, f=f: _FEAT_VALS[f][i])
            for f in range(self._n_features)
        ]
        _SHARED_G = gradients
        _SHARED_H = hessians

        self._nodes = []
        try:
            if n_workers > 1:
                ctx = multiprocessing.get_context("fork")
                self._pool = ctx.Pool(processes=n_workers)
            else:
                self._pool = None

            self._build(
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

    def get_leaves(self) -> List[float]:
        if not self._is_fit:
            raise RuntimeError("Tree has not been fit yet")
        return [node.leaf_value for node in self._nodes if node.is_leaf]

    def set_leaves(self, values: List[float]) -> None:
        leaf_nodes = [node for node in self._nodes if node.is_leaf]
        if len(values) != len(leaf_nodes):
            raise ValueError(
                f"set_leaves() got {len(values)} values "
                f"but tree has {len(leaf_nodes)} leaves"
            )
        vi = 0
        for node in self._nodes:
            if node.is_leaf:
                node.leaf_value = float(values[vi])
                vi += 1

    def n_leaves(self) -> int:
        return sum(1 for n in self._nodes if n.is_leaf)

    def n_nodes(self) -> int:
        return len(self._nodes)

    def depth(self) -> int:
        if not self._nodes:
            return 0
        if not any(node.is_split() for node in self._nodes):
            return 0
        return max(
            int(math.floor(math.log2(i + 1)))
            for i, node in enumerate(self._nodes)
            if node.is_split()
        )

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