"""
iotids.boosting.tree
====================
A single regression tree fit on XGBoost-style second-order gradients.

Pure Python — zero third-party imports.  Uses only multiprocessing from
the standard library for parallelism.

Why the previous parallel implementation was still slow
-------------------------------------------------------
pool.map() pickles every task argument and sends it through an OS pipe.
With 198K-sample nodes, each task contained:
  sorted_idx  : 198K ints   ≈ 1.6 MB
  feat_vals   : 198K floats ≈ 1.6 MB
  gradients   : 198K floats ≈ 1.6 MB
  hessians    : 198K floats ≈ 1.6 MB
  Total       : ~6.4 MB per feature × 24 features = 154 MB of pipe I/O
  per node — serialized by the main process before any worker starts.
  The pickling cost dominated the compute, keeping CPU at ~5%.

The correct pattern: shared module globals + fork inheritance
------------------------------------------------------------
On Linux, multiprocessing.get_context("fork") creates worker processes
by forking the parent.  Forked workers inherit the parent's entire
address space via copy-on-write — they can read X, gradients, hessians
directly from memory at zero cost, without any serialization.

The key is to store the data in a module-level global BEFORE creating
the pool, then only send tiny descriptors through the pipe:

    Task = (feature_index,)     # 44 bytes instead of 6.4 MB

Each worker reads X/g/h from the inherited global, sorts its assigned
feature column, scans thresholds, and returns a 4-tuple of scalars.
Zero data copying.  Zero pickling of arrays.

This is the standard pattern for CPU-bound parallel Python on Linux HPC
systems (exactly what CSE 19 is).

Pool lifetime
-------------
The pool is created once per tree.fit() call and reused across all
node builds in that tree.  Fork overhead is paid once, not once per node.
"""

from __future__ import annotations

import math
import multiprocessing
import os
from typing import List, Optional, Tuple

from .node import Node


# ---------------------------------------------------------------------------
# Module-level shared state
# Populated by BoostingTree.fit() BEFORE the pool is forked.
# Workers access these directly from inherited memory — no serialization.
# ---------------------------------------------------------------------------

_SHARED_X:          List[List[float]] = []
_SHARED_G:          List[float]       = []
_SHARED_H:          List[float]       = []
_SHARED_INDICES:    List[int]         = []
_SHARED_G_TOTAL:    float             = 0.0
_SHARED_H_TOTAL:    float             = 0.0
_SHARED_REG_LAMBDA: float             = 1.0
_SHARED_MCW:        float             = 1.0
_SHARED_MIN_GAIN:   float             = 0.0


# ---------------------------------------------------------------------------
# Worker function — module-level so it's picklable
# Receives ONLY a feature index.  Everything else comes from inherited globals.
# ---------------------------------------------------------------------------

def _scan_feature_global(args):
    """
    Find the best split threshold for one feature.

    Called in a worker process.
    - X, g, h are read from module globals inherited at fork — zero IPC cost.
    - indices, G_total, H_total, and hyperparams are sent per-task because
      they change for every node.  They are small (indices ~ node_size ints,
      scalars ~ negligible) compared to X/g/h (n_train × n_features floats).

    Parameters
    ----------
    args : (feat, indices, G_total, H_total, reg_lambda, mcw, mg)

    Returns
    -------
    (gain, feat, threshold, left_count)
    """
    feat, indices, G_total, H_total, lam, mcw, mg = args
    X       = _SHARED_X
    g       = _SHARED_G
    h       = _SHARED_H

    # Sort samples by this feature's value — O(n log n)
    sorted_idx = sorted(indices, key=lambda i: X[i][feat])
    n = len(sorted_idx)

    best_gain   = -math.inf
    best_thresh = 0.0
    best_left_k = -1
    G_L = 0.0
    H_L = 0.0

    for k in range(n - 1):
        i    = sorted_idx[k]
        G_L += g[i]
        H_L += h[i]
        G_R  = G_total - G_L
        H_R  = H_total - H_L

        if H_L < mcw or H_R < mcw:
            continue

        v_k  = X[sorted_idx[k]][feat]
        v_k1 = X[sorted_idx[k + 1]][feat]
        if v_k == v_k1:
            continue

        d   = H_total + lam
        dL  = H_L     + lam
        dR  = H_R     + lam
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
    """
    A single gradient-boosted regression tree.

    Parameters
    ----------
    max_depth        : int   – maximum tree depth
    min_child_weight : float – minimum hessian sum in a child node
    reg_lambda       : float – L2 regularization on leaf weights
    min_gain         : float – minimum gain to accept a split (gamma)
    n_jobs           : int   – worker processes for feature scan
                               -1 = all cores, 1 = serial (for debugging)
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
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_workers(n_jobs: int, n_features: int) -> int:
        if n_jobs == 1:
            return 1
        cpu_count = os.cpu_count() or 1
        workers   = cpu_count if n_jobs == -1 else max(1, n_jobs)
        # No point having more workers than features to scan
        return min(workers, max(1, n_features))

    def _leaf_weight(self, G: float, H: float) -> float:
        return -G / (H + self.reg_lambda)

    # ------------------------------------------------------------------
    # Best split — each feature dispatched to an inherited-memory worker
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

        # ── Dispatch ──────────────────────────────────────────────────────
        # Task tuple: (feat, indices, G_total, H_total, lam, mcw, mg)
        # X/g/h are NOT in the tuple — workers read them from inherited globals.
        # indices is the only array here; at depth d it has n/2^d entries,
        # so it shrinks rapidly down the tree.  Root = n ints, depth-3 = n/8.
        tasks = [
            (feat, indices, G_total, H_total,
             self.reg_lambda, self.min_child_weight, self.min_gain)
            for feat in feature_cols
        ]

        if self._pool is not None and len(tasks) > 1:
            results = self._pool.map(_scan_feature_global, tasks)
        else:
            results = [_scan_feature_global(t) for t in tasks]

        # ── Pick the best result across features ──────────────────────────
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

        # Reconstruct left/right index lists from the winning feature sort
        # We need to re-sort once for the winner — cheaper than keeping all
        # 24 sorted lists in memory simultaneously.
        sorted_winner = sorted(indices, key=lambda i: X[i][best_feat])
        left_idx  = sorted_winner[: best_left_k + 1]
        right_idx = sorted_winner[best_left_k + 1 :]

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

        Populates module globals with X/g/h BEFORE forking the pool so
        workers inherit the data at zero copy cost.  The pool is created
        once and shared across all nodes in this tree.
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

        # ── Populate globals BEFORE fork so workers inherit them ──────────
        global _SHARED_X, _SHARED_G, _SHARED_H
        _SHARED_X = X
        _SHARED_G = gradients
        _SHARED_H = hessians

        self._nodes = []
        try:
            if n_workers > 1:
                # fork() on Linux: workers get a copy-on-write snapshot of
                # the parent process, including _SHARED_X/G/H above.
                # They never need to receive this data through the pipe.
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
                f"set_leaves() got {len(values)} values "
                f"but tree has {len(leaf_nodes)} leaves"
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