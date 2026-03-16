"""
CIC-IDS-2017  XGBoost Baseline  —  iotids Library
==================================================
Trains and evaluates an XGBoostClassifier using only the iotids
boosting/ library. No external ML dependencies (no xgboost package,
no scikit-learn, no TensorFlow, no pandas).

Environment (CSE 19)
--------------------
    source /scratch/yjh5327/venv311/bin/activate
    cd /scratch/yjh5327/CMPSC496
    python3 -u xgboost.py --data processed_reduced/clean_dataset.csv

Default run (10% sample, ~283K rows, ~30-60 min):
    nohup python3 -u xg_boost.py --data processed_reduced/clean_dataset.csv \
        > logs/xgb_run.log 2>&1 &
    tail -f logs/xgb_run.log

Full dataset (2.8M rows — expect several hours, pure Python):
    nohup python3 -u xg_boost.py --data processed_reduced/clean_dataset.csv \
        --sample 1.0 > logs/xgb_full.log 2>&1 &

Expected results (full dataset, default hyperparams):
    Accuracy  : ~99%
    Precision : ~99%
    Recall    : ~97%
    F1        : ~98%
    AUC       : ~0.999

Dataset
-------
    processed_reduced/clean_dataset.csv
    2,830,743 rows — 24 numeric features + Label column
    Class balance: 80.3% BENIGN, 19.7% attack

Federated note
--------------
    XGBoostClassifier exposes get_weights() / set_weights() / clone() /
    local_train(seed=round_seed) — the full FedAvg contract required by
    federated/client.py and federated/server.py.  This script validates
    centralised training accuracy before plugging into the FL loop.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

# ---------------------------------------------------------------------------
# Path setup — allows running from /scratch/yjh5327/CMPSC496 with iotids/
# sitting alongside this file or one directory up.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in [_SCRIPT_DIR, os.path.dirname(_SCRIPT_DIR)]:
    if os.path.isdir(os.path.join(_candidate, "iotids")):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        break

# iotids imports — library only, zero external ML dependencies
from iotids.boosting import XGBoostClassifier, save_xgb
from iotids.boosting.xgboost_classifier import _logloss
from iotids.data.csv_reader import read_csv
from iotids.data.preprocessing import (
    RobustScaler,
    clip_outliers,
    drop_nan_rows,
    replace_inf,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # ── Data ────────────────────────────────────────────────────────────────
    DATA_FILE       = "processed_reduced/clean_dataset.csv"
    SAMPLE_FRAC     = 0.1           # 0.1 = ~283K rows, completes in ~30-60 min
                                    # set --sample 1.0 for the full 2.8M row run

    # ── Split ────────────────────────────────────────────────────────────────
    TRAIN_FRAC      = 0.70
    VAL_FRAC        = 0.15
    # TEST_FRAC is implicit: 1 - TRAIN - VAL = 0.15

    # ── XGBoost hyperparams ──────────────────────────────────────────────────
    # These were tuned for CIC-IDS-2017 with 24 features.
    # Full derivation:
    #   n_estimators=200:  enough rounds to converge; early stopping cuts it down
    #   learning_rate=0.1: standard XGBoost default; smaller = more rounds needed
    #   max_depth=6:       XGBoost default; deeper causes overfitting on CIC-IDS
    #   min_child_weight=5: slightly higher than default (1) to prevent tiny
    #                       splits on rare attack sub-classes
    #   subsample=0.8:     row subsampling — standard noise injection
    #   colsample=0.8:     column subsampling — with only 24 features, 0.8
    #                      keeps ~19 features per tree (sqrt would be 5)
    #   reg_lambda=1.0:    XGBoost default L2
    #   reg_alpha=0.1:     light L1 helps with the noisy CIC-IDS features
    #   min_gain=0.0:      no hard pruning; early stopping handles over-fitting
    N_ESTIMATORS        = 200
    LEARNING_RATE       = 0.1
    MAX_DEPTH           = 6
    MIN_CHILD_WEIGHT    = 5.0
    SUBSAMPLE           = 0.8
    COLSAMPLE_BYTREE    = 0.8
    REG_LAMBDA          = 1.0
    REG_ALPHA           = 0.1
    MIN_GAIN            = 0.0
    EARLY_STOPPING      = 15       # rounds without val_logloss improvement

    # ── Threshold ────────────────────────────────────────────────────────────
    DECISION_THRESHOLD  = 0.5      # updated by threshold_sweep on validation set
    OPTIMIZE_THRESHOLD  = True

    # ── Oversampling ─────────────────────────────────────────────────────────
    # Oversampling the minority (attack) class in the training split helps
    # gradient boosting converge faster when the class imbalance is severe.
    # CIC-IDS-2017 is 80/20 — light oversampling (0.4 ratio) is usually enough.
    USE_OVERSAMPLING    = True
    OVERSAMPLE_RATIO    = 0.4      # target attack/benign ratio after oversample

    # ── Output ───────────────────────────────────────────────────────────────
    SAVE_MODEL          = True
    MODEL_OUT           = "models/xgb_baseline.iotids"
    RESULTS_OUT         = "results/xgb_results.txt"

    # ── Columns to exclude from feature matrix ───────────────────────────────
    EXCLUDE_COLS = {
        "Label", " Label",
        "Flow ID", " Flow ID",
        "Source IP", " Source IP",
        "Destination IP", " Destination IP",
        "Timestamp", " Timestamp",
        "Source Port", " Source Port",
        "Destination Port", " Destination Port",
    }

    RANDOM_SEED = 42


# ============================================================================
# HELPERS
# ============================================================================

def _banner(title: str, width: int = 80) -> None:
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}m {s}s"


def _stratified_split(
    X: list, y: list, train_frac: float, val_frac: float, seed: int
) -> tuple:
    """
    Stratified 3-way split into train / val / test.

    Preserves the class ratio in every split by splitting each class
    separately and then interleaving.  Identical approach to rf.py and dnn.py
    so results are directly comparable.
    """
    rng = random.Random(seed)

    # Separate indices by class
    pos_idx = [i for i, yi in enumerate(y) if yi == 1]
    neg_idx = [i for i, yi in enumerate(y) if yi == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    def _split_class(idx):
        n      = len(idx)
        n_tr   = int(math.ceil(n * train_frac))
        n_va   = int(math.ceil(n * val_frac))
        return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    pos_tr, pos_va, pos_te = _split_class(pos_idx)
    neg_tr, neg_va, neg_te = _split_class(neg_idx)

    def _gather(ix):
        xs = [X[i] for i in ix]
        ys = [float(y[i]) for i in ix]
        combined = list(zip(xs, ys))
        rng.shuffle(combined)
        xs, ys = zip(*combined) if combined else ([], [])
        return list(xs), list(ys)

    X_tr, y_tr = _gather(pos_tr + neg_tr)
    X_va, y_va = _gather(pos_va + neg_va)
    X_te, y_te = _gather(pos_te + neg_te)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def _oversample(
    X: list, y: list, ratio: float, seed: int
) -> tuple:
    """
    Randomly duplicate minority (attack, y=1) samples until
    attack / benign ≈ ratio.

    Only applied to the training split to prevent data leakage.
    """
    rng = random.Random(seed)
    pos = [(X[i], y[i]) for i in range(len(y)) if y[i] == 1]
    neg = [(X[i], y[i]) for i in range(len(y)) if y[i] == 0]

    n_neg = len(neg)
    target_pos = int(n_neg * ratio)

    if target_pos <= len(pos):
        # Already at or above target ratio — no oversampling needed
        combined = pos + neg
    else:
        extra = target_pos - len(pos)
        augmented = pos + [rng.choice(pos) for _ in range(extra)]
        combined = augmented + neg

    rng.shuffle(combined)
    X_out = [c[0] for c in combined]
    y_out = [c[1] for c in combined]
    return X_out, y_out


# ============================================================================
# LOADING
# ============================================================================

def load_data(config: Config) -> dict:
    _banner("LOADING CIC-IDS-2017 DATASET")

    path = config.DATA_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n  Dataset not found: {path}\n"
            f"  Pass the correct path via --data <path>"
        )

    print(f"\n  Loading: {path}")
    t0 = time.time()
    data = read_csv(path)
    elapsed = time.time() - t0

    n_rows = len(next(iter(data.values())))
    n_cols = len(data)
    print(f"  Loaded {n_rows:,} rows, {n_cols} columns  ({_fmt_time(elapsed)})")

    # Class distribution (before preprocessing)
    label_key = None
    for k in data:
        if k.strip() == "Label":
            label_key = k
            break
    if label_key is None:
        raise KeyError("No 'Label' column found.  Check the CSV header.")

    labels = data[label_key]
    counts: dict = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    total = sum(counts.values())
    print("  Class distribution:")
    for lbl, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {str(lbl):<40} {cnt:>9,}  ({cnt / total * 100:.2f}%)")

    return data


# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess(data: dict, config: Config) -> tuple:
    _banner("DATA PREPROCESSING")

    # Identify label column
    label_key = next(k for k in data if k.strip() == "Label")

    feat_names = [
        k for k in data
        if k not in config.EXCLUDE_COLS and k.strip() not in config.EXCLUDE_COLS
    ]
    print(f"\n  Feature columns  : {len(feat_names)}")

    # Convert to float
    print("  Converting to float arrays...")
    n_rows = len(data[feat_names[0]])
    X_raw = []
    for i in range(n_rows):
        row = []
        for col in feat_names:
            try:
                row.append(float(data[col][i]))
            except (ValueError, TypeError):
                row.append(float("nan"))
        X_raw.append(row)

    raw_labels = list(data[label_key])

    # Clean
    print("  Replacing Inf values...")
    X_raw = replace_inf(X_raw)
    print("  Dropping NaN rows...")
    X_raw, raw_labels = drop_nan_rows(X_raw, raw_labels)
    print("  Clipping outliers [1st, 99th pct]...")
    X_raw = clip_outliers(X_raw, low_pct=1, high_pct=99)

    # Binary labels: 0 = BENIGN, 1 = attack
    # Handles both:
    #   processed_reduced/clean_dataset.csv  -> numeric  "0.0" / "1.0"
    #   raw CIC-IDS-2017 CSVs               -> string   "BENIGN" / attack name
    def _to_binary(lbl) -> int:
        s = str(lbl).strip()
        try:
            return 0 if float(s) == 0.0 else 1   # numeric labels
        except ValueError:
            return 0 if s == "BENIGN" else 1       # string labels

    # Diagnose before encoding so bad labels are visible immediately
    raw_unique = list(set(str(l).strip() for l in raw_labels))
    print(f"  Raw label values (up to 10): {raw_unique[:10]}")

    y = [_to_binary(lbl) for lbl in raw_labels]

    n_total  = len(y)
    n_benign = sum(1 for v in y if v == 0)
    n_attack = sum(1 for v in y if v == 1)

    print(f"\n  After cleaning:")
    print(f"    Samples  : {n_total:,}")
    print(f"    Features : {len(feat_names)}")
    print(f"    Benign   : {n_benign:,}  ({n_benign / n_total * 100:.1f}%)")
    print(f"    Attack   : {n_attack:,}  ({n_attack / n_total * 100:.1f}%)")

    return X_raw, y, feat_names


# ============================================================================
# SAMPLING
# ============================================================================

def maybe_sample(
    X: list, y: list, frac: float, seed: int
) -> tuple:
    if frac >= 1.0:
        return X, y
    rng = random.Random(seed)
    n = len(y)
    k = max(1, int(math.ceil(frac * n)))

    # Stratified sampling: preserve class ratio
    pos_idx = [i for i in range(n) if y[i] == 1]
    neg_idx = [i for i in range(n) if y[i] == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    k_pos = max(1, int(round(k * len(pos_idx) / n)))
    k_neg = k - k_pos
    if k_neg > len(neg_idx):
        k_neg = len(neg_idx)

    chosen = pos_idx[:k_pos] + neg_idx[:k_neg]
    rng.shuffle(chosen)

    Xs = [X[i] for i in chosen]
    ys = [y[i] for i in chosen]

    pos_rate = sum(ys) / max(1, len(ys))
    print(f"  Stratified {frac * 100:.0f}% sample: {len(ys):,} rows  "
          f"(attack rate {pos_rate * 100:.1f}%)")
    return Xs, ys


# ============================================================================
# SPLIT + SCALE
# ============================================================================

def split_and_scale(X: list, y: list, config: Config) -> dict:
    _banner("DATA SPLITTING AND SCALING")

    X_tr, y_tr, X_va, y_va, X_te, y_te = _stratified_split(
        X, y, config.TRAIN_FRAC, config.VAL_FRAC, seed=config.RANDOM_SEED
    )

    print(f"  Train : {len(X_tr):>9,}  ({config.TRAIN_FRAC * 100:.0f}%)")
    print(f"  Val   : {len(X_va):>9,}  ({config.VAL_FRAC * 100:.0f}%)")
    te_pct = (1.0 - config.TRAIN_FRAC - config.VAL_FRAC) * 100
    print(f"  Test  : {len(X_te):>9,}  ({te_pct:.0f}%)")

    print("\n  Fitting RobustScaler on training data...")
    scaler = RobustScaler()
    scaler.fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)
    print("  Scaling complete.")

    # Oversample training split only
    if config.USE_OVERSAMPLING:
        n_before = len(y_tr)
        n_attack_before = sum(y_tr)
        X_tr, y_tr = _oversample(X_tr, y_tr, config.OVERSAMPLE_RATIO, config.RANDOM_SEED)
        n_after  = len(y_tr)
        n_attack_after = int(sum(y_tr))
        print(f"\n  Oversampling (ratio={config.OVERSAMPLE_RATIO}):")
        print(f"    Before : {n_before:,}  (attack={n_attack_before:,})")
        print(f"    After  : {n_after:,}  (attack={n_attack_after:,})")

    return {
        "X_tr": X_tr, "y_tr": y_tr,
        "X_va": X_va, "y_va": y_va,
        "X_te": X_te, "y_te": y_te,
        "scaler": scaler,
    }


# ============================================================================
# TRAINING
# ============================================================================

def train(splits: dict, config: Config) -> XGBoostClassifier:
    _banner("TRAINING XGBoost  (iotids.boosting)")

    X_tr, y_tr = splits["X_tr"], splits["y_tr"]
    X_va, y_va = splits["X_va"], splits["y_va"]

    print(f"  n_estimators     : {config.N_ESTIMATORS}")
    print(f"  learning_rate    : {config.LEARNING_RATE}")
    print(f"  max_depth        : {config.MAX_DEPTH}")
    print(f"  min_child_weight : {config.MIN_CHILD_WEIGHT}")
    print(f"  subsample        : {config.SUBSAMPLE}")
    print(f"  colsample_bytree : {config.COLSAMPLE_BYTREE}")
    print(f"  reg_lambda       : {config.REG_LAMBDA}")
    print(f"  reg_alpha        : {config.REG_ALPHA}")
    print(f"  early_stopping   : {config.EARLY_STOPPING} rounds")
    print(f"  Training on      : {len(X_tr):,} samples  "
          f"({len(X_tr[0])} features)")
    print()

    clf = XGBoostClassifier(
        n_estimators=config.N_ESTIMATORS,
        learning_rate=config.LEARNING_RATE,
        max_depth=config.MAX_DEPTH,
        min_child_weight=config.MIN_CHILD_WEIGHT,
        subsample=config.SUBSAMPLE,
        colsample_bytree=config.COLSAMPLE_BYTREE,
        reg_lambda=config.REG_LAMBDA,
        reg_alpha=config.REG_ALPHA,
        min_gain=config.MIN_GAIN,
        early_stopping_rounds=config.EARLY_STOPPING,
        seed=config.RANDOM_SEED,
    )

    t0 = time.time()
    clf.fit(
        X_tr,
        [float(yi) for yi in y_tr],
        eval_set=(X_va, [float(yi) for yi in y_va]),
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\n  Training complete in {_fmt_time(elapsed)}")
    print(f"  Trees built      : {clf._best_n_trees}")

    return clf


# ============================================================================
# THRESHOLD OPTIMISATION
# ============================================================================

def optimise_threshold(
    clf: XGBoostClassifier,
    X_va: list,
    y_va: list,
    config: Config,
) -> float:
    if not config.OPTIMIZE_THRESHOLD:
        return config.DECISION_THRESHOLD

    _banner("THRESHOLD OPTIMISATION  (val set)")
    best_t = clf.threshold_sweep(X_va, [float(yi) for yi in y_va], metric="f1")
    print(f"  Optimal threshold (max-F1 on val) : {best_t:.4f}")

    # Also print what happens at a few candidate thresholds
    probs = clf.predict_proba(X_va)
    y_val_f = [float(yi) for yi in y_va]
    print(f"\n  {'Threshold':>10}  {'Accuracy':>10}  {'Precision':>10}  "
          f"{'Recall':>10}  {'F1':>10}")
    for t in [0.3, 0.4, 0.5, best_t, 0.6, 0.7]:
        preds = [1 if p >= t else 0 for p in probs]
        tp = sum(1 for yp, yt in zip(preds, y_va) if yp == 1 and yt == 1)
        fp = sum(1 for yp, yt in zip(preds, y_va) if yp == 1 and yt == 0)
        fn = sum(1 for yp, yt in zip(preds, y_va) if yp == 0 and yt == 1)
        tn = sum(1 for yp, yt in zip(preds, y_va) if yp == 0 and yt == 0)
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1   = 2 * prec * rec / max(1e-9, prec + rec)
        acc  = (tp + tn) / max(1, len(y_va))
        marker = " <-- optimal" if abs(t - best_t) < 1e-6 else ""
        print(f"  {t:>10.4f}  {acc:>10.4f}  {prec:>10.4f}  "
              f"{rec:>10.4f}  {f1:>10.4f}{marker}")

    return best_t


# ============================================================================
# EVALUATION
# ============================================================================

def _roc_auc(y_true: list, y_prob: list) -> float:
    """Trapezoidal AUC — no numpy required."""
    paired = sorted(zip(y_prob, y_true), key=lambda x: -x[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = 0
    fp = 0
    auc = 0.0
    prev_tp = 0
    prev_fp = 0

    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        # Trapezoidal rule
        auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
        prev_tp = tp
        prev_fp = fp

    return auc / (n_pos * n_neg)


def evaluate(
    clf: XGBoostClassifier,
    splits: dict,
    feat_names: list,
    config: Config,
    threshold: float,
) -> dict:
    _banner("EVALUATION")

    results = {}
    for split_name, (X, y) in [
        ("Validation", (splits["X_va"], splits["y_va"])),
        ("Test",       (splits["X_te"], splits["y_te"])),
    ]:
        y_f = [float(yi) for yi in y]
        probs = clf.predict_proba(X)
        preds = [1 if p >= threshold else 0 for p in probs]

        tp = sum(1 for yp, yt in zip(preds, y) if yp == 1 and yt == 1)
        fp = sum(1 for yp, yt in zip(preds, y) if yp == 1 and yt == 0)
        fn = sum(1 for yp, yt in zip(preds, y) if yp == 0 and yt == 1)
        tn = sum(1 for yp, yt in zip(preds, y) if yp == 0 and yt == 0)

        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1   = 2 * prec * rec / max(1e-9, prec + rec)
        acc  = (tp + tn) / max(1, len(y))
        auc  = _roc_auc(y_f, probs)
        ll   = _logloss(y_f, probs)

        results[split_name] = {
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "auc":       auc,
            "logloss":   ll,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_samples": len(y),
        }

        print(f"\n  ── {split_name} ({len(y):,} samples) ──")
        print(f"  Accuracy   : {acc:.4f}  ({acc * 100:.2f}%)")
        print(f"  Precision  : {prec:.4f}")
        print(f"  Recall     : {rec:.4f}")
        print(f"  F1 Score   : {f1:.4f}")
        print(f"  AUC-ROC    : {auc:.4f}")
        print(f"  Log-Loss   : {ll:.4f}")
        print(f"  Confusion matrix (threshold={threshold:.4f}):")
        print(f"    TP={tp:>8,}  FP={fp:>8,}")
        print(f"    FN={fn:>8,}  TN={tn:>8,}")

    return results


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def report_feature_importance(
    clf: XGBoostClassifier, feat_names: list, top_n: int = 15
) -> None:
    _banner(f"FEATURE IMPORTANCE  (top {top_n})")

    importances = clf.feature_importances_
    ranked = sorted(
        zip(feat_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    bar_width = 30
    print(f"\n  {'Rank':>4}  {'Feature':<40}  {'Importance':>10}  {'Bar'}")
    print(f"  {'----':>4}  {'-------':<40}  {'----------':>10}  {'---'}")
    for rank, (name, imp) in enumerate(ranked[:top_n], 1):
        bar_len = int(round(imp * bar_width / max(ranked[0][1], 1e-9)))
        bar = "█" * bar_len
        print(f"  {rank:>4}  {name.strip():<40}  {imp:>10.6f}  {bar}")


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(
    results: dict,
    feat_names: list,
    config: Config,
    threshold: float,
    n_trees: int,
    elapsed_train: float,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(config.RESULTS_OUT)), exist_ok=True)
    with open(config.RESULTS_OUT, "w") as f:
        f.write("XGBoost Baseline — iotids Library — CIC-IDS-2017\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Trees built        : {n_trees}\n")
        f.write(f"Features           : {len(feat_names)}\n")
        f.write(f"Decision threshold : {threshold:.4f}\n")
        f.write(f"Training time      : {_fmt_time(elapsed_train)}\n\n")
        for split_name, m in results.items():
            f.write(f"{split_name} Results\n")
            f.write("-" * 40 + "\n")
            for k, v in m.items():
                if isinstance(v, float):
                    f.write(f"  {k:<15}: {v:.6f}\n")
                else:
                    f.write(f"  {k:<15}: {v}\n")
            f.write("\n")
    print(f"\n  Results saved to: {config.RESULTS_OUT}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="XGBoost baseline for CIC-IDS-2017 — iotids library only"
    )
    parser.add_argument(
        "--data", default=Config.DATA_FILE,
        help="Path to the CIC-IDS-2017 CSV file "
             "(default: processed_reduced/clean_dataset.csv)"
    )
    parser.add_argument(
        "--sample", type=float, default=Config.SAMPLE_FRAC,
        metavar="FRAC",
        help="Fraction of dataset to use, e.g. 0.1 for a quick test "
             "(default: 1.0 = full dataset)"
    )
    parser.add_argument(
        "--n-estimators", type=int, default=Config.N_ESTIMATORS,
        help=f"Number of boosting rounds (default: {Config.N_ESTIMATORS})"
    )
    parser.add_argument(
        "--max-depth", type=int, default=Config.MAX_DEPTH,
        help=f"Maximum tree depth (default: {Config.MAX_DEPTH})"
    )
    parser.add_argument(
        "--lr", type=float, default=Config.LEARNING_RATE,
        help=f"Learning rate / shrinkage (default: {Config.LEARNING_RATE})"
    )
    parser.add_argument(
        "--no-oversample", action="store_true",
        help="Disable minority-class oversampling on the training split"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving the trained model to disk"
    )
    parser.add_argument(
        "--seed", type=int, default=Config.RANDOM_SEED,
        help=f"Random seed (default: {Config.RANDOM_SEED})"
    )
    args = parser.parse_args()

    config = Config()
    config.DATA_FILE        = args.data
    config.SAMPLE_FRAC      = args.sample
    config.N_ESTIMATORS     = args.n_estimators
    config.MAX_DEPTH        = args.max_depth
    config.LEARNING_RATE    = args.lr
    config.RANDOM_SEED      = args.seed
    config.USE_OVERSAMPLING = not args.no_oversample
    if args.no_save:
        config.SAVE_MODEL = False

    # ── Header ────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("XGBoost BASELINE  —  iotids Library".center(80))
    print("No TensorFlow  ·  No xgboost package  ·  No scikit-learn  ·  No pandas".center(80))
    print("=" * 80)

    random.seed(config.RANDOM_SEED)

    # ── Pipeline ──────────────────────────────────────────────────────────
    t_start = time.time()

    # 1. Load
    data = load_data(config)

    # 2. Preprocess
    X, y, feat_names = preprocess(data, config)

    # 3. Sample (optional)
    if config.SAMPLE_FRAC < 1.0:
        _banner("SAMPLING")
        X, y = maybe_sample(X, y, config.SAMPLE_FRAC, config.RANDOM_SEED)

    # 4. Split + scale
    splits = split_and_scale(X, y, config)

    # 5. Train
    t_train_start = time.time()
    clf = train(splits, config)
    t_train_elapsed = time.time() - t_train_start

    # 6. Threshold optimisation
    threshold = optimise_threshold(
        clf, splits["X_va"], splits["y_va"], config
    )
    clf.set_threshold(threshold)

    # 7. Evaluate
    results = evaluate(clf, splits, feat_names, config, threshold)

    # 8. Feature importance
    report_feature_importance(clf, feat_names, top_n=15)

    # 9. Save model
    if config.SAVE_MODEL:
        _banner("SAVING MODEL")
        os.makedirs(os.path.dirname(os.path.abspath(config.MODEL_OUT)), exist_ok=True)
        save_xgb(clf, config.MODEL_OUT)
        size_kb = os.path.getsize(config.MODEL_OUT) / 1024
        print(f"  Model saved : {config.MODEL_OUT}  ({size_kb:.1f} KB)")

    # 10. Save results text file
    save_results(
        results, feat_names, config, threshold,
        n_trees=clf._best_n_trees,
        elapsed_train=t_train_elapsed,
    )

    # ── Summary ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_start
    test_m = results.get("Test", {})
    _banner("SUMMARY")
    print(f"\n  Dataset        : {config.DATA_FILE}")
    print(f"  Sample frac    : {config.SAMPLE_FRAC}")
    print(f"  Trees built    : {clf._best_n_trees}")
    print(f"  Threshold      : {threshold:.4f}")
    print()
    print(f"  Test accuracy  : {test_m.get('accuracy',  0):.4f}  "
          f"({test_m.get('accuracy',  0) * 100:.2f}%)")
    print(f"  Test precision : {test_m.get('precision', 0):.4f}")
    print(f"  Test recall    : {test_m.get('recall',    0):.4f}")
    print(f"  Test F1        : {test_m.get('f1',        0):.4f}")
    print(f"  Test AUC-ROC   : {test_m.get('auc',       0):.4f}")
    print(f"  Test log-loss  : {test_m.get('logloss',   0):.4f}")
    print()
    print(f"  Total runtime  : {_fmt_time(total_elapsed)}")
    print(f"  Training time  : {_fmt_time(t_train_elapsed)}")
    print()

    # Comparison note
    print("  Comparison (same test split):")
    print("    DNN  baseline : ~99.42% accuracy  (reference)")
    print("    RF   baseline : ~99.65% accuracy  (reference)")
    print(f"    XGB  baseline : {test_m.get('accuracy', 0) * 100:.2f}% accuracy  (this run)")
    print()


if __name__ == "__main__":
    main()