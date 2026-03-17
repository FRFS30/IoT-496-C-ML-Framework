"""
iotids.boosting.tree  (compact storage edition)
================================================
Switches _build() from heap indexing to compact sequential DFS storage.

Old layout (heap): node at depth d, position p occupies index 2^d + p - 1.
  A depth-5 tree with early stops can have max_index=89 with only 37 real
  nodes -- 58% wasted space filled with make_empty() sentinels.

New layout (compact): nodes appended in pre-order DFS traversal order.
  left_child / right_child store actual list indices.
  37 real nodes occupy exactly indices 0..36 -- zero padding.

predict_one() updated to follow left_child/right_child instead of 2*i+1/2*i+2.
All other public API (fit, predict, get_leaves, set_leaves, to_dict) unchanged.
Serializer backward compatible: new fields in node.to_dict() / from_dict().
"""

from __future__ import annotations

import math
import multiprocessing
import os
from typing import List, Optional, Tuple

from .node import Node


_SORTED_COLS: List[List[int]]   = []
_SHARED_G:    List[float]       = []
_SHARED_H:    List[float]       = []
_FEAT_VALS:   List[List[float]] = []


def _scan_feature_global(args):
    feat, active_set, G_total, H_total, lam, mcw, mg = args
    sorted_all = _SORTED_COLS[feat]
    feat_vals  = _FEAT_VALS[feat]
    g          = _SHARED_G
    h          = _SHARED_H

    sorted_idx = [i for i in sorted_all if i in active_set]
    n = len(sorted_idx)
    if n < 2:
        return -math.inf, feat, 0.0, -1

    best_gain = -math.inf
    best_thresh = 0.0
    best_left_k = -1
    G_L = H_L = 0.0

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
        gain = 0.5*(G_L*G_L/(H_L+lam) + G_R*G_R/(H_R+lam) - G_total*G_total/d) - mg
        if gain > best_gain:
            best_gain, best_thresh, best_left_k = gain, (v_k+v_k1)*0.5, k

    return best_gain, feat, best_thresh, best_left_k


class BoostingTree:

    def __init__(self, max_depth=6, min_child_weight=1.0,
                 reg_lambda=1.0, min_gain=0.0, n_jobs=-1):
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
    def _resolve_workers(n_jobs, n_features):
        if n_jobs == 1:
            return 1
        cpu_count = os.cpu_count() or 1
        workers   = cpu_count if n_jobs == -1 else max(1, n_jobs)
        return min(workers, max(1, n_features))

    def _leaf_weight(self, G, H):
        return -G / (H + self.reg_lambda)

    def _best_split(self, indices, feature_cols, G_total, H_total):
        if len(indices) < 2:
            return -1, 0.0, -math.inf, [], []

        active_set = frozenset(indices)
        lam = self.reg_lambda
        mcw = self.min_child_weight
        mg  = self.min_gain

        tasks = [(feat, active_set, G_total, H_total, lam, mcw, mg)
                 for feat in feature_cols]

        if self._pool is not None:
            results = self._pool.map(_scan_feature_global, tasks)
        else:
            results = [_scan_feature_global(t) for t in tasks]

        best_gain, best_feat, best_thresh, best_left_k = max(results, key=lambda r: r[0])

        if best_feat == -1 or best_gain <= 0.0 or best_left_k < 0:
            return -1, 0.0, -math.inf, [], []

        sorted_winner = [i for i in _SORTED_COLS[best_feat] if i in active_set]
        return (best_feat, best_thresh, best_gain,
                sorted_winner[:best_left_k+1], sorted_winner[best_left_k+1:])

    def _build(self, indices, feature_cols, depth):
        """
        Compact DFS builder. Appends nodes to self._nodes sequentially.
        Returns the index of the node just created.
        """
        g = _SHARED_G
        h = _SHARED_H

        G = sum(g[i] for i in indices)
        H = sum(h[i] for i in indices)
        n = len(indices)

        node_idx = len(self._nodes)

        if depth >= self.max_depth or n == 0:
            self._nodes.append(Node.make_leaf(
                self._leaf_weight(G, H), sample_count=n, sum_hessian=H))
            return node_idx

        best_feat, best_thresh, best_gain, left_idx, right_idx = \
            self._best_split(indices, feature_cols, G, H)

        if best_feat == -1 or best_gain <= 0.0:
            self._nodes.append(Node.make_leaf(
                self._leaf_weight(G, H), sample_count=n, sum_hessian=H))
            return node_idx

        # Reserve slot for this split node, fill in child indices after recursion
        split_node = Node.make_split(
            feature_idx=best_feat, threshold=best_thresh,
            gain=best_gain, sample_count=n, sum_hessian=H)
        self._nodes.append(split_node)

        left_child  = self._build(left_idx,  feature_cols, depth+1)
        right_child = self._build(right_idx, feature_cols, depth+1)

        # Now set the child indices on the split node
        self._nodes[node_idx].left_child  = left_child
        self._nodes[node_idx].right_child = right_child

        return node_idx

    def fit(self, X, gradients, hessians, feature_cols=None):
        n_samples = len(X)
        if n_samples == 0:
            raise ValueError("fit() received empty dataset")
        if len(gradients) != n_samples or len(hessians) != n_samples:
            raise ValueError("Length mismatch: X, gradients, hessians")

        self._n_features = len(X[0]) if X else 0
        if feature_cols is None:
            feature_cols = list(range(self._n_features))

        n_workers = self._resolve_workers(self.n_jobs, len(feature_cols))

        global _SORTED_COLS, _SHARED_G, _SHARED_H, _FEAT_VALS
        _FEAT_VALS   = [[X[i][f] for i in range(n_samples)] for f in range(self._n_features)]
        _SORTED_COLS = [sorted(range(n_samples), key=lambda i, f=f: _FEAT_VALS[f][i])
                        for f in range(self._n_features)]
        _SHARED_G = gradients
        _SHARED_H = hessians

        self._nodes = []
        try:
            if n_workers > 1:
                ctx = multiprocessing.get_context("fork")
                self._pool = ctx.Pool(processes=n_workers)
            else:
                self._pool = None
            self._build(list(range(n_samples)), feature_cols, depth=0)
        finally:
            if self._pool is not None:
                self._pool.close()
                self._pool.join()
                self._pool = None

        self._is_fit = True
        return self

    def predict_one(self, x):
        if not self._is_fit:
            raise RuntimeError("Tree has not been fit yet")
        node = self._nodes[0]
        while not node.is_leaf:
            go_left = x[node.feature_idx] <= node.threshold
            child_idx = node.left_child if go_left else node.right_child
            node = self._nodes[child_idx]
        return node.leaf_value

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def get_leaves(self):
        if not self._is_fit:
            raise RuntimeError("Tree has not been fit yet")
        return [n.leaf_value for n in self._nodes if n.is_leaf]

    def set_leaves(self, values):
        leaf_nodes = [n for n in self._nodes if n.is_leaf]
        if len(values) != len(leaf_nodes):
            raise ValueError(f"set_leaves() got {len(values)} values "
                             f"but tree has {len(leaf_nodes)} leaves")
        vi = 0
        for n in self._nodes:
            if n.is_leaf:
                n.leaf_value = float(values[vi])
                vi += 1

    def n_leaves(self):
        return sum(1 for n in self._nodes if n.is_leaf)

    def n_nodes(self):
        return len(self._nodes)

    def depth(self):
        if not self._nodes:
            return 0
        # BFS to find max depth
        from collections import deque
        q = deque([(0, 0)])
        max_d = 0
        while q:
            idx, d = q.popleft()
            max_d = max(max_d, d)
            node = self._nodes[idx]
            if not node.is_leaf:
                q.append((node.left_child, d+1))
                q.append((node.right_child, d+1))
        return max_d

    def feature_gains(self, n_features=None):
        nf = n_features if n_features is not None else self._n_features
        gains = [0.0] * nf
        for node in self._nodes:
            if not node.is_leaf and 0 <= node.feature_idx < nf:
                gains[node.feature_idx] += node.gain
        return gains

    def to_dict(self):
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
    def from_dict(cls, d):
        tree = cls(max_depth=d["max_depth"],
                   min_child_weight=d["min_child_weight"],
                   reg_lambda=d["reg_lambda"],
                   min_gain=d["min_gain"])
        tree._n_features = d["n_features"]
        tree._is_fit     = d["is_fit"]
        tree._nodes      = [Node.from_dict(nd) for nd in d["nodes"]]
        return tree

    def __repr__(self):
        if not self._is_fit:
            return f"BoostingTree(max_depth={self.max_depth}, [not fit])"
        return (f"BoostingTree(max_depth={self.max_depth}, "
                f"depth={self.depth()}, n_nodes={self.n_nodes()}, "
                f"n_leaves={self.n_leaves()})")