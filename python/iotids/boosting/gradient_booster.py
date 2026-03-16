"""
iotids.boosting.gradient_booster
=================================
Core gradient boosting engine: fits a sequence of BoostingTree objects on
first- and second-order gradient residuals to minimize binary cross-entropy.

Mathematical overview
---------------------
We minimize the regularized log-loss objective across N samples and T trees:

    F_t(x) = F_{t-1}(x) + eta * f_t(x)      (eta = learning_rate)

where f_t is the t-th regression tree fit on:

    g_i = d L / d F_{t-1}(x_i)  =  sigma(F_{t-1}(x_i)) - y_i
    h_i = d^2 L / d F_{t-1}(x_i)^2  =  sigma(F_{t-1}(x_i)) * (1 - sigma(F_{t-1}(x_i)))

Initial raw score F_0 is the log-odds of the training positive rate:

    F_0 = log( mean(y) / (1 - mean(y)) )

Predictions are produced as:

    P(y=1 | x) = sigma( F_T(x) )  =  1 / (1 + exp(-F_T(x)))

Regularization
--------------
* L2 (reg_lambda) : folded into each tree's leaf weight formula via BoostingTree
* L1 (reg_alpha)  : applied as soft-thresholding on leaf values after FedAvg
                    weight updates (prevents leaf explosion after averaging)
* subsample       : row subsampling — random fraction of training samples per tree
* colsample_bytree: column subsampling — random fraction of features per tree

All randomness is seeded through a single integer seed that can be set per
round by the federated server (round_seed), ensuring all clients build
structurally identical trees for leaf-value averaging to be valid.

Feature importance
------------------
feature_importances_[j] = total gain accumulated at split nodes using feature j,
summed across all T trees.  Normalized to sum to 1.0.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

from .tree import BoostingTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    # Avoid exp overflow for large negative x
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _log_odds(p: float, eps: float = 1e-7) -> float:
    """Safe log-odds: log(p / (1-p)) with clipping to avoid ±inf."""
    p = max(eps, min(1.0 - eps, p))
    return math.log(p / (1.0 - p))


def _logloss(y_true: List[float], y_prob: List[float], eps: float = 1e-7) -> float:
    """Binary cross-entropy for validation monitoring."""
    total = 0.0
    for yt, yp in zip(y_true, y_prob):
        yp = max(eps, min(1.0 - eps, yp))
        total += -(yt * math.log(yp) + (1.0 - yt) * math.log(1.0 - yp))
    return total / max(1, len(y_true))


# ---------------------------------------------------------------------------
# GradientBooster
# ---------------------------------------------------------------------------

class GradientBooster:
    """
    Gradient boosting classifier for binary classification.

    This class orchestrates the sequential tree-fitting loop.  It is not meant
    to be used directly in most cases — use XGBoostClassifier for the full
    sklearn-compatible API with early stopping and FedAvg integration.

    Parameters
    ----------
    n_estimators     : int   – number of boosting rounds (trees)
    learning_rate    : float – shrinkage applied to each tree's leaf values (eta)
    max_depth        : int   – maximum depth per tree
    min_child_weight : float – minimum hessian sum in a child (XGBoost param)
    subsample        : float – row subsampling fraction per tree (0 < s <= 1)
    colsample_bytree : float – column subsampling fraction per tree (0 < c <= 1)
    reg_lambda       : float – L2 regularization on leaf weights
    reg_alpha        : float – L1 soft-thresholding on leaf values post-FedAvg
    min_gain         : float – minimum gain threshold (XGBoost gamma)
    base_score       : float – initial probability estimate; defaults to 0.5
                               (overridden by training label mean when fit)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        min_gain: float = 0.0,
        base_score: float = 0.5,
        n_jobs: int = -1,
    ) -> None:
        # Validate hyperparameters
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0.0 < subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1]")
        if not 0.0 < colsample_bytree <= 1.0:
            raise ValueError("colsample_bytree must be in (0, 1]")
        if reg_lambda < 0:
            raise ValueError("reg_lambda must be >= 0")
        if reg_alpha < 0:
            raise ValueError("reg_alpha must be >= 0")

        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.max_depth         = max_depth
        self.min_child_weight  = min_child_weight
        self.subsample         = subsample
        self.colsample_bytree  = colsample_bytree
        self.reg_lambda        = reg_lambda
        self.reg_alpha         = reg_alpha
        self.min_gain          = min_gain
        self.base_score        = base_score
        self.n_jobs            = n_jobs

        # State populated by fit()
        self._trees:          List[BoostingTree] = []
        self._F0:             float = 0.0    # initial raw score (log-odds of base_score)
        self._n_features:     int   = 0
        self._is_fit:         bool  = False
        self._feature_gains:  List[float] = []  # accumulated per-feature gains

    # ------------------------------------------------------------------
    # Internal: gradient computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_gradients(
        y: List[float], raw_scores: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        Compute first- and second-order gradient statistics.

        For binary log-loss:
          g_i = sigma(F(x_i)) - y_i
          h_i = sigma(F(x_i)) * (1 - sigma(F(x_i)))

        Both g and h are needed by BoostingTree to compute split gains and
        optimal leaf weights.
        """
        gradients = []
        hessians  = []
        for yi, fi in zip(y, raw_scores):
            pi = _sigmoid(fi)
            gradients.append(pi - yi)
            hessians.append(max(pi * (1.0 - pi), 1e-7))   # floor avoids /0
        return gradients, hessians

    # ------------------------------------------------------------------
    # Internal: subsampling
    # ------------------------------------------------------------------

    def _row_sample(self, n: int, rng: random.Random) -> List[int]:
        """Return a sorted list of row indices after subsampling."""
        if self.subsample >= 1.0:
            return list(range(n))
        k = max(1, int(math.ceil(self.subsample * n)))
        return sorted(rng.sample(range(n), k))

    def _col_sample(self, n_features: int, rng: random.Random) -> List[int]:
        """Return a sorted list of column indices after subsampling."""
        if self.colsample_bytree >= 1.0:
            return list(range(n_features))
        k = max(1, int(math.ceil(self.colsample_bytree * n_features)))
        return sorted(rng.sample(range(n_features), k))

    # ------------------------------------------------------------------
    # Public: fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: List[List[float]],
        y: List[float],
        seed: int = 42,
        eval_X: Optional[List[List[float]]] = None,
        eval_y: Optional[List[float]] = None,
        verbose: bool = False,
    ) -> "GradientBooster":
        """
        Fit gradient boosted trees on training data.

        Parameters
        ----------
        X       : List[List[float]] – (n_samples, n_features) training matrix
        y       : List[float]       – binary labels (0 or 1)
        seed    : int               – controls all randomness in subsampling;
                                      must match across FL clients each round
        eval_X  : optional validation features for logloss monitoring
        eval_y  : optional validation labels
        verbose : print per-round metrics

        Returns self.
        """
        n_samples  = len(X)
        n_features = len(X[0]) if X else 0

        if n_samples == 0 or n_features == 0:
            raise ValueError("Training data is empty")
        if len(y) != n_samples:
            raise ValueError(f"X has {n_samples} rows but y has {len(y)} elements")

        self._n_features    = n_features
        self._trees         = []
        self._feature_gains = [0.0] * n_features

        # Base score: log-odds of training positive rate
        pos_rate = sum(y) / max(1, n_samples)
        self._F0 = _log_odds(pos_rate if pos_rate > 0 else self.base_score)

        # Current raw scores for every sample
        raw_scores = [self._F0] * n_samples

        rng = random.Random(seed)

        for t in range(self.n_estimators):
            # ── Compute gradients on full training set ─────────────
            gradients, hessians = self._compute_gradients(y, raw_scores)

            # ── Subsampling ────────────────────────────────────────
            row_idx = self._row_sample(n_samples, rng)
            col_idx = self._col_sample(n_features, rng)

            X_sub  = [X[i]          for i in row_idx]
            g_sub  = [gradients[i]  for i in row_idx]
            h_sub  = [hessians[i]   for i in row_idx]

            # ── Build tree on subsampled residuals ─────────────────
            tree = BoostingTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                min_gain=self.min_gain,
                n_jobs=self.n_jobs,
            )
            tree.fit(X_sub, g_sub, h_sub, feature_cols=col_idx)

            # ── Update raw scores on full training set ─────────────
            tree_preds = tree.predict(X)
            for i in range(n_samples):
                raw_scores[i] += self.learning_rate * tree_preds[i]

            # ── Accumulate feature gains ───────────────────────────
            fg = tree.feature_gains(n_features)
            for j in range(n_features):
                self._feature_gains[j] += fg[j]

            self._trees.append(tree)

            # ── Optional verbose logging ───────────────────────────
            if verbose:
                train_probs = [_sigmoid(s) for s in raw_scores]
                train_ll    = _logloss(y, train_probs)
                msg = f"[{t+1:>4}/{self.n_estimators}]  train_logloss={train_ll:.6f}"
                if eval_X is not None and eval_y is not None:
                    val_probs = self.predict_proba(eval_X)
                    val_ll    = _logloss(eval_y, val_probs)
                    msg += f"  val_logloss={val_ll:.6f}"
                print(msg)

        self._is_fit = True
        return self

    # ------------------------------------------------------------------
    # Public: prediction
    # ------------------------------------------------------------------

    def _raw_score(self, x: List[float]) -> float:
        """Accumulated raw score for a single sample across all trees."""
        score = self._F0
        for tree in self._trees:
            score += self.learning_rate * tree.predict_one(x)
        return score

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        """
        Return P(y=1 | x) for each sample in X.

        Output is in [0, 1].  Pass through threshold for binary labels.
        """
        if not self._is_fit:
            raise RuntimeError("Model has not been fit yet")
        return [_sigmoid(self._raw_score(x)) for x in X]

    def predict(self, X: List[List[float]], threshold: float = 0.5) -> List[int]:
        """Return binary predictions using the given probability threshold."""
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

    def evaluate(
        self,
        X: List[List[float]],
        y: List[float],
        threshold: float = 0.5,
    ) -> dict:
        """
        Compute classification metrics on (X, y).

        Returns a dict with keys: accuracy, logloss, n_samples.
        More detailed metrics are computed by metrics/classification.py —
        this is a lightweight self-contained check used during training.
        """
        probs = self.predict_proba(X)
        preds = [1 if p >= threshold else 0 for p in probs]
        correct = sum(1 for yp, yt in zip(preds, y) if yp == int(yt))
        return {
            "accuracy":  correct / max(1, len(y)),
            "logloss":   _logloss(y, probs),
            "n_samples": len(y),
        }

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    @property
    def feature_importances_(self) -> List[float]:
        """
        Normalized per-feature gain importance.

        Returns a list of length n_features summing to 1.0.
        Features never used in any split have importance 0.0.
        """
        if not self._is_fit:
            raise RuntimeError("Model has not been fit yet")
        total = sum(self._feature_gains)
        if total == 0.0:
            return [0.0] * self._n_features
        return [g / total for g in self._feature_gains]

    # ------------------------------------------------------------------
    # FedAvg weight interface
    # ------------------------------------------------------------------

    def get_weights(self) -> List[List[float]]:
        """
        Return leaf values for all trees.

        Shape: list of T lists, each containing the leaf values of one tree.
        The federated server averages corresponding entries across clients.
        """
        if not self._is_fit:
            raise RuntimeError("Model has not been fit yet")
        return [tree.get_leaves() for tree in self._trees]

    def set_weights(self, weights: List[List[float]]) -> None:
        """
        Replace leaf values with federated-averaged values from the server.

        Optionally applies L1 soft-thresholding (reg_alpha) to prevent leaf
        value explosion after averaging — without this, multiple rounds of
        FedAvg can slowly inflate leaf magnitudes.

        Parameters
        ----------
        weights : List[List[float]] – one leaf list per tree, same structure
                                      as returned by get_weights()
        """
        if len(weights) != len(self._trees):
            raise ValueError(
                f"set_weights() got {len(weights)} weight arrays "
                f"but model has {len(self._trees)} trees"
            )
        for tree, leaf_vals in zip(self._trees, weights):
            if self.reg_alpha > 0.0:
                # Soft-threshold: sign(w) * max(|w| - alpha, 0)
                leaf_vals = [
                    math.copysign(max(abs(v) - self.reg_alpha, 0.0), v)
                    for v in leaf_vals
                ]
            tree.set_leaves(leaf_vals)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "n_estimators":      self.n_estimators,
            "learning_rate":     self.learning_rate,
            "max_depth":         self.max_depth,
            "min_child_weight":  self.min_child_weight,
            "subsample":         self.subsample,
            "colsample_bytree":  self.colsample_bytree,
            "reg_lambda":        self.reg_lambda,
            "reg_alpha":         self.reg_alpha,
            "min_gain":          self.min_gain,
            "base_score":        self.base_score,
            "F0":                self._F0,
            "n_features":        self._n_features,
            "is_fit":            self._is_fit,
            "feature_gains":     self._feature_gains,
            "trees":             [t.to_dict() for t in self._trees],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GradientBooster":
        obj = cls(
            n_estimators=d["n_estimators"],
            learning_rate=d["learning_rate"],
            max_depth=d["max_depth"],
            min_child_weight=d["min_child_weight"],
            subsample=d["subsample"],
            colsample_bytree=d["colsample_bytree"],
            reg_lambda=d["reg_lambda"],
            reg_alpha=d["reg_alpha"],
            min_gain=d["min_gain"],
            base_score=d["base_score"],
        )
        obj._F0             = d["F0"]
        obj._n_features     = d["n_features"]
        obj._is_fit         = d["is_fit"]
        obj._feature_gains  = d["feature_gains"]
        obj._trees          = [BoostingTree.from_dict(t) for t in d["trees"]]
        return obj

    def __repr__(self) -> str:
        status = f"fit, {len(self._trees)} trees" if self._is_fit else "not fit"
        return (
            f"GradientBooster(n_estimators={self.n_estimators}, "
            f"lr={self.learning_rate}, max_depth={self.max_depth}, "
            f"subsample={self.subsample}, colsample={self.colsample_bytree}, "
            f"[{status}])"
        )