"""
iotids.boosting.xgboost_classifier
===================================
Scikit-learn-compatible XGBoost classifier wrapping GradientBooster.

This is the primary interface for the federated framework.  It adds:

  * Early stopping based on validation log-loss
  * Threshold calibration via threshold_sweep()
  * The full FedAvg contract: get_weights() / set_weights() / clone()
  * round_seed propagation for structurally identical trees across FL clients
  * get_params() / set_params() for hyperparameter serialization
  * A clean evaluate() dict matching nn/model.py for uniform reporting

Federated contract
------------------
All three model families (nn/, forest/, boosting/) expose:

    get_weights() -> opaque weight structure
    set_weights(weights)  -- in-place replacement; no retraining required
    clone()               -- deep copy for per-client local model instances

For boosting/, weights are List[List[float]] — one leaf-value list per tree.
FedAvgServer averages them element-wise across clients.

The round_seed constraint
-------------------------
XGBoost FedAvg leaf averaging is only valid when all clients build trees with
the same node structure.  This is enforced by the server broadcasting a
`round_seed` at the start of each FL round.  Clients call:

    client.model.local_train(X, y, seed=round_seed)

which passes the seed to GradientBooster.fit() and therefore to all row/column
subsampling calls — guaranteeing identical tree topology across clients.

Threshold calibration
---------------------
INT8 quantization shifts the DNN's optimal threshold (0.70 -> 0.35).
Similarly, XGBoost's sigmoid output may have a different optimal threshold
than the DNN, especially post-FedAvg.  Call threshold_sweep() after each
training stage to find the threshold that maximises F1 on the validation set,
then pass it to predict() and evaluate().
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

from .gradient_booster import GradientBooster, _logloss, _sigmoid


# ---------------------------------------------------------------------------
# Threshold sweep helper (mirrors metrics/classification.py API)
# ---------------------------------------------------------------------------

def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _precision_recall_at_threshold(
    y_true: List[float], y_prob: List[float], threshold: float
) -> Tuple[float, float]:
    tp = fp = fn = 0
    for yt, yp in zip(y_true, y_prob):
        pred = 1 if yp >= threshold else 0
        true = int(yt)
        if pred == 1 and true == 1:
            tp += 1
        elif pred == 1 and true == 0:
            fp += 1
        elif pred == 0 and true == 1:
            fn += 1
    precision = tp / max(1, tp + fp)
    recall    = tp / max(1, tp + fn)
    return precision, recall


# ---------------------------------------------------------------------------
# XGBoostClassifier
# ---------------------------------------------------------------------------

class XGBoostClassifier:
    """
    XGBoost binary classifier with FedAvg integration.

    Parameters
    ----------
    n_estimators            : int   – number of boosting rounds
    learning_rate           : float – shrinkage (eta)
    max_depth               : int   – maximum tree depth
    min_child_weight        : float – minimum hessian sum in child nodes
    subsample               : float – row subsampling per round (0, 1]
    colsample_bytree        : float – column subsampling per round (0, 1]
    reg_lambda              : float – L2 leaf regularization
    reg_alpha               : float – L1 leaf soft-thresholding (post-FedAvg)
    min_gain                : float – minimum gain to accept a split (gamma)
    base_score              : float – initial probability estimate
    early_stopping_rounds   : int   – stop if val_logloss doesn't improve for
                                      this many rounds (0 = disabled)
    seed                    : int   – default random seed for fit(); overridden
                                      per FL round by local_train(seed=round_seed)
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
        early_stopping_rounds: int = 10,
        seed: int = 42,
    ) -> None:
        self.n_estimators           = n_estimators
        self.learning_rate          = learning_rate
        self.max_depth              = max_depth
        self.min_child_weight       = min_child_weight
        self.subsample              = subsample
        self.colsample_bytree       = colsample_bytree
        self.reg_lambda             = reg_lambda
        self.reg_alpha              = reg_alpha
        self.min_gain               = min_gain
        self.base_score             = base_score
        self.early_stopping_rounds  = early_stopping_rounds
        self.seed                   = seed

        self._booster: Optional[GradientBooster] = None
        self._best_n_trees: int = 0
        self._optimal_threshold: float = 0.5
        self._is_fit: bool = False

    # ------------------------------------------------------------------
    # Internal: build a fresh GradientBooster from current params
    # ------------------------------------------------------------------

    def _make_booster(self, n_estimators: Optional[int] = None) -> GradientBooster:
        return GradientBooster(
            n_estimators=n_estimators if n_estimators is not None else self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            min_gain=self.min_gain,
            base_score=self.base_score,
        )

    # ------------------------------------------------------------------
    # Public: fit with optional early stopping
    # ------------------------------------------------------------------

    def fit(
        self,
        X: List[List[float]],
        y: List[float],
        eval_set: Optional[Tuple[List[List[float]], List[float]]] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> "XGBoostClassifier":
        """
        Train the model.

        Parameters
        ----------
        X        : training features
        y        : binary training labels (0 / 1 as float)
        eval_set : optional (X_val, y_val) tuple for early stopping and
                   threshold calibration
        seed     : overrides self.seed for this call; used by FL clients to
                   pass round_seed without mutating the model's default seed
        verbose  : print per-round loss

        With early stopping
        -------------------
        If eval_set is provided and early_stopping_rounds > 0, training runs
        tree-by-tree: a fresh GradientBooster with n_estimators=1 is used in a
        loop so we can monitor val_logloss each round without re-training from
        scratch.  When val_logloss has not improved for early_stopping_rounds
        consecutive rounds, training stops and the best checkpoint is restored.

        Without early stopping (or no eval_set)
        ----------------------------------------
        A single GradientBooster.fit() call with the full n_estimators.
        Significantly faster — preferred when n_estimators is already tuned.
        """
        effective_seed = seed if seed is not None else self.seed

        if eval_set is not None and self.early_stopping_rounds > 0:
            X_val, y_val = eval_set
            self._fit_with_early_stopping(
                X, y, X_val, y_val, effective_seed, verbose
            )
        else:
            booster = self._make_booster()
            booster.fit(
                X, y,
                seed=effective_seed,
                eval_X=eval_set[0] if eval_set else None,
                eval_y=eval_set[1] if eval_set else None,
                verbose=verbose,
            )
            self._booster = booster
            self._best_n_trees = len(booster._trees)

        self._is_fit = True

        # Calibrate threshold if validation data is available
        if eval_set is not None:
            X_val, y_val = eval_set
            self._optimal_threshold = self.threshold_sweep(X_val, y_val)

        return self

    def _fit_with_early_stopping(
        self,
        X: List[List[float]],
        y: List[float],
        X_val: List[List[float]],
        y_val: List[float],
        seed: int,
        verbose: bool,
    ) -> None:
        """
        Incremental fitting loop with early stopping.

        Builds one tree per iteration by fitting a temporary single-tree
        booster on the current residuals, then appending the tree manually.
        This is the standard way to implement early stopping in a pure-Python
        XGBoost from scratch.
        """
        import random as _random
        rng = _random.Random(seed)

        # Build a full booster but stop it manually
        booster = self._make_booster(n_estimators=self.n_estimators)
        booster._n_features = len(X[0]) if X else 0
        booster._trees = []
        booster._feature_gains = [0.0] * booster._n_features

        # Compute base score
        pos_rate = sum(y) / max(1, len(y))
        from .gradient_booster import _log_odds
        booster._F0 = _log_odds(pos_rate if pos_rate > 0 else booster.base_score)

        raw_scores = [booster._F0] * len(X)

        best_val_ll  = math.inf
        best_trees   = []
        best_fg      = [0.0] * booster._n_features
        no_improve   = 0

        for t in range(self.n_estimators):
            # Gradients
            g, h = GradientBooster._compute_gradients(y, raw_scores)

            # Subsampling — use the round sub-RNG so each tree gets a fresh state
            row_idx = booster._row_sample(len(X), rng)
            col_idx = booster._col_sample(booster._n_features, rng)

            X_sub = [X[i] for i in row_idx]
            g_sub = [g[i] for i in row_idx]
            h_sub = [h[i] for i in row_idx]

            from .tree import BoostingTree
            tree = BoostingTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                min_gain=self.min_gain,
            )
            tree.fit(X_sub, g_sub, h_sub, feature_cols=col_idx)

            # Update raw scores
            tree_preds = tree.predict(X)
            for i in range(len(X)):
                raw_scores[i] += booster.learning_rate * tree_preds[i]

            # Accumulate gains
            fg = tree.feature_gains(booster._n_features)
            for j in range(booster._n_features):
                booster._feature_gains[j] += fg[j]

            booster._trees.append(tree)

            # Validation check
            val_probs = [_sigmoid(booster._F0 + sum(
                booster.learning_rate * booster._trees[k].predict_one(xv)
                for k in range(len(booster._trees))
            )) for xv in X_val]
            val_ll = _logloss(y_val, val_probs)

            if verbose:
                train_probs = [_sigmoid(s) for s in raw_scores]
                train_ll = _logloss(y, train_probs)
                print(f"[{t+1:>4}/{self.n_estimators}]  "
                      f"train_logloss={train_ll:.6f}  val_logloss={val_ll:.6f}")

            if val_ll < best_val_ll - 1e-7:
                best_val_ll  = val_ll
                best_trees   = [copy.deepcopy(tr) for tr in booster._trees]
                best_fg      = list(booster._feature_gains)
                no_improve   = 0
            else:
                no_improve += 1
                if no_improve >= self.early_stopping_rounds:
                    if verbose:
                        print(f"Early stopping at round {t+1}. "
                              f"Best val_logloss={best_val_ll:.6f} at round "
                              f"{t+1 - no_improve}")
                    break

        booster._trees          = best_trees
        booster._feature_gains  = best_fg
        booster._is_fit         = True
        self._booster           = booster
        self._best_n_trees      = len(best_trees)

    # ------------------------------------------------------------------
    # Public: prediction
    # ------------------------------------------------------------------

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        """Return P(y=1 | x) for each sample."""
        self._check_fit()
        return self._booster.predict_proba(X)

    def predict(
        self,
        X: List[List[float]],
        threshold: Optional[float] = None,
    ) -> List[int]:
        """
        Return binary predictions.

        Parameters
        ----------
        threshold : float | None – uses self._optimal_threshold if None
                    (set by threshold_sweep during fit with eval_set, or
                    manually via set_threshold())
        """
        t = threshold if threshold is not None else self._optimal_threshold
        return [1 if p >= t else 0 for p in self.predict_proba(X)]

    def evaluate(
        self,
        X: List[List[float]],
        y: List[float],
        threshold: Optional[float] = None,
    ) -> dict:
        """
        Evaluate on (X, y) and return a metrics dict.

        Returns
        -------
        dict with keys: accuracy, precision, recall, f1, logloss,
                        n_samples, threshold_used
        """
        self._check_fit()
        t = threshold if threshold is not None else self._optimal_threshold
        probs = self.predict_proba(X)
        preds = [1 if p >= t else 0 for p in probs]

        tp = fp = fn = tn = 0
        for yp, yt in zip(preds, y):
            true = int(yt)
            if yp == 1 and true == 1:
                tp += 1
            elif yp == 1 and true == 0:
                fp += 1
            elif yp == 0 and true == 1:
                fn += 1
            else:
                tn += 1

        precision = tp / max(1, tp + fp)
        recall    = tp / max(1, tp + fn)
        accuracy  = (tp + tn) / max(1, len(y))

        return {
            "accuracy":       accuracy,
            "precision":      precision,
            "recall":         recall,
            "f1":             _f1(precision, recall),
            "logloss":        _logloss(y, probs),
            "n_samples":      len(y),
            "threshold_used": t,
            "n_trees":        len(self._booster._trees),
        }

    # ------------------------------------------------------------------
    # Threshold calibration
    # ------------------------------------------------------------------

    def threshold_sweep(
        self,
        X_val: List[List[float]],
        y_val: List[float],
        n_thresholds: int = 100,
        metric: str = "f1",
    ) -> float:
        """
        Find the probability threshold that maximizes `metric` on the
        validation set.

        Parameters
        ----------
        X_val        : validation features
        y_val        : validation labels
        n_thresholds : number of candidate thresholds to evaluate
        metric       : 'f1' | 'accuracy'

        Returns
        -------
        best_threshold : float – also stored in self._optimal_threshold
        """
        self._check_fit()
        probs = self.predict_proba(X_val)
        thresholds = [i / n_thresholds for i in range(1, n_thresholds)]

        best_t     = 0.5
        best_score = -1.0

        for t in thresholds:
            if metric == "f1":
                prec, rec = _precision_recall_at_threshold(y_val, probs, t)
                score = _f1(prec, rec)
            elif metric == "accuracy":
                correct = sum(1 for yt, yp in zip(y_val, probs)
                              if (1 if yp >= t else 0) == int(yt))
                score = correct / max(1, len(y_val))
            else:
                raise ValueError(f"Unknown metric '{metric}'. Use 'f1' or 'accuracy'.")

            if score > best_score:
                best_score = score
                best_t     = t

        self._optimal_threshold = best_t
        return best_t

    def set_threshold(self, threshold: float) -> None:
        """Manually set the decision threshold."""
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Threshold must be in (0, 1), got {threshold}")
        self._optimal_threshold = threshold

    # ------------------------------------------------------------------
    # FedAvg contract
    # ------------------------------------------------------------------

    def get_weights(self) -> List[List[float]]:
        """
        Return leaf value arrays for all trees.

        Shape: List[T] of List[n_leaves_t].
        Used by FedAvgServer to average across clients.
        """
        self._check_fit()
        return self._booster.get_weights()

    def set_weights(self, weights: List[List[float]]) -> None:
        """
        Replace leaf values with federated-averaged values.

        reg_alpha soft-thresholding is applied inside GradientBooster.set_weights()
        if reg_alpha > 0.
        """
        self._check_fit()
        self._booster.set_weights(weights)

    def local_train(
        self,
        X: List[List[float]],
        y: List[float],
        seed: int,
        eval_set: Optional[Tuple[List[List[float]], List[float]]] = None,
        verbose: bool = False,
    ) -> "XGBoostClassifier":
        """
        Re-train this model on local client data using the given round_seed.

        Called by FederatedClient.local_train() each FL round.  The seed is
        broadcast by FedAvgServer to enforce identical tree structure across
        all clients.

        Returns self for chaining.
        """
        return self.fit(X, y, eval_set=eval_set, seed=seed, verbose=verbose)

    def clone(self) -> "XGBoostClassifier":
        """
        Return a deep copy of this classifier.

        Used by FederatedClient to create per-client local model instances
        from the global model without sharing state.
        """
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    @property
    def feature_importances_(self) -> List[float]:
        """Normalized per-feature gain importance from GradientBooster."""
        self._check_fit()
        return self._booster.feature_importances_

    def top_features(
        self, n: int = 20, feature_names: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Return the top-n most important features as (name, importance) pairs.

        Parameters
        ----------
        n             : number of features to return
        feature_names : optional list of column names; uses 'feat_i' if None
        """
        importances = self.feature_importances_
        names = feature_names if feature_names else [f"feat_{i}" for i in range(len(importances))]
        ranked = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
        return ranked[:n]

    # ------------------------------------------------------------------
    # Hyperparameter interface (mirrors sklearn convention)
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        """Return all hyperparameters as a dict."""
        return {
            "n_estimators":           self.n_estimators,
            "learning_rate":          self.learning_rate,
            "max_depth":              self.max_depth,
            "min_child_weight":       self.min_child_weight,
            "subsample":              self.subsample,
            "colsample_bytree":       self.colsample_bytree,
            "reg_lambda":             self.reg_lambda,
            "reg_alpha":              self.reg_alpha,
            "min_gain":               self.min_gain,
            "base_score":             self.base_score,
            "early_stopping_rounds":  self.early_stopping_rounds,
            "seed":                   self.seed,
        }

    def set_params(self, **params: Any) -> "XGBoostClassifier":
        """Set hyperparameters.  Does not re-train."""
        valid = set(self.get_params().keys())
        for k, v in params.items():
            if k not in valid:
                raise ValueError(f"Unknown parameter '{k}'")
            setattr(self, k, v)
        return self

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "params":              self.get_params(),
            "optimal_threshold":   self._optimal_threshold,
            "best_n_trees":        self._best_n_trees,
            "is_fit":              self._is_fit,
            "booster":             self._booster.to_dict() if self._booster else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "XGBoostClassifier":
        obj = cls(**d["params"])
        obj._optimal_threshold = d["optimal_threshold"]
        obj._best_n_trees      = d["best_n_trees"]
        obj._is_fit            = d["is_fit"]
        if d["booster"] is not None:
            obj._booster = GradientBooster.from_dict(d["booster"])
        return obj

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fit(self) -> None:
        if not self._is_fit or self._booster is None:
            raise RuntimeError(
                "XGBoostClassifier has not been fit yet. Call fit() first."
            )

    def __repr__(self) -> str:
        status = (
            f"fit, {self._best_n_trees} trees, threshold={self._optimal_threshold:.3f}"
            if self._is_fit else "not fit"
        )
        return (
            f"XGBoostClassifier("
            f"n_estimators={self.n_estimators}, "
            f"lr={self.learning_rate}, "
            f"max_depth={self.max_depth}, "
            f"[{status}])"
        )
