"""
CIC-IDS-2017 Random Forest Baseline — iotids Library
=====================================================
Trains and evaluates a Random Forest classifier using only the iotids
Python library. No TensorFlow, scikit-learn, or pandas dependency.

Expected Result: ~99% accuracy on the CIC-IDS-2017 test set.

Usage:
    python rf.py
    python rf.py --data path/to/clean_dataset.csv
    python rf.py --sample 0.1              # 10% stratified sample
    python rf.py --sample 0.1 --repeats 5  # 5 independent 10% runs -> mean±std
"""

import argparse
import json
import math
import pickle
import time
from pathlib import Path

# ── iotids imports only ───────────────────────────────────────────────────────
from python.iotids.data.csv_reader    import read_csv
from python.iotids.data.preprocessing import (
    RobustScaler,
    LabelEncoder,
    clip_outliers,
    replace_inf,
    drop_nan_rows,
)
from python.iotids.data.dataset        import Dataset
from python.iotids.forest.random_forest import RandomForestClassifier
from python.iotids.forest.serializer   import save_rf, load_rf
from python.iotids.metrics.classification import (
    accuracy,
    precision,
    recall,
    f1_score,
    roc_auc,
    confusion_matrix,
    threshold_sweep,
    classification_report,
)
from python.iotids.utils.random import set_seed
from python.iotids.utils.io     import save, load
from python.iotids.utils.math   import percentile
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    DATA_FILE      = Path("processed_reduced/clean_dataset.csv")
    OUTPUT_DIR     = Path("results/iotids_rf")
    MODEL_DIR      = Path("models/iotids_rf")

    # Dataset
    USE_SAMPLE     = False
    SAMPLE_FRAC    = 0.10
    REPEATS        = 1          # number of independent sample+train+eval runs

    # Model hyperparameters
    N_ESTIMATORS       = 100
    MAX_DEPTH          = 20
    MIN_SAMPLES_SPLIT  = 50
    MIN_SAMPLES_LEAF   = 20
    MAX_FEATURES       = "sqrt"

    # Split sizes
    TEST_SIZE       = 0.20
    VALIDATION_SIZE = 0.10

    # Non-feature columns in CIC-IDS-2017
    EXCLUDE_COLS = {
        "Label", "Flow_ID", "Source_IP", "Destination_IP",
        "Timestamp", "Source_Port", "Destination_Port",
    }


# ============================================================================
# DATA LOADING  (load once — sampling happens per repeat)
# ============================================================================

def load_data(config: Config):
    """Load full CIC-IDS-2017 via iotids csv_reader. Sampling done per repeat."""
    print("\n" + "=" * 80)
    print("LOADING CIC-IDS-2017 DATASET")
    print("=" * 80)

    if not config.DATA_FILE.exists():
        raise FileNotFoundError(f"\n  Dataset not found: {config.DATA_FILE}")

    print(f"\n  Loading: {config.DATA_FILE}")
    data = read_csv(str(config.DATA_FILE))

    n_rows = len(next(iter(data.values())))
    print(f"  Loaded {n_rows:,} rows, {len(data)} columns")

    labels_raw = data["Label"]
    label_counts: dict = {}
    for lbl in labels_raw:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    print(f"\n  Class distribution:")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {str(lbl):<35} {cnt:>10,}  ({cnt / n_rows * 100:>5.2f}%)")

    return data


def _stratified_sample(data: dict, frac: float, seed: int) -> dict:
    """Return a stratified random fraction of the raw column dict."""
    import random as _rng
    _rng.seed(seed)

    labels = data["Label"]
    class_indices: dict = {}
    for i, lbl in enumerate(labels):
        class_indices.setdefault(lbl, []).append(i)

    selected = []
    for idxs in class_indices.values():
        k = max(1, int(len(idxs) * frac))
        selected.extend(_rng.sample(idxs, k))

    selected_set = set(selected)
    return {col: [v for i, v in enumerate(vals) if i in selected_set]
            for col, vals in data.items()}


# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess(data: dict, config: Config):
    feature_cols = [
        col for col in data
        if col not in config.EXCLUDE_COLS
        and all(
            isinstance(v, (int, float)) or
            (isinstance(v, str) and _is_numeric(v))
            for v in data[col][:100]
        )
    ]

    n_rows = len(next(iter(data.values())))

    X_cols = []
    for col in feature_cols:
        col_vals = []
        for v in data[col]:
            try:
                f = float(v)
            except (ValueError, TypeError):
                f = float("nan")
            col_vals.append(f)
        X_cols.append(col_vals)

    X = [[X_cols[j][i] for j in range(len(feature_cols))]
         for i in range(n_rows)]

    benign = {"BENIGN", "Benign", "benign"}
    raw_labels = data["Label"]
    y = []
    for lbl in raw_labels:
        try:
            y.append(int(lbl))
        except (ValueError, TypeError):
            y.append(0 if str(lbl) in benign else 1)

    X = replace_inf(X)
    X, y = drop_nan_rows(X, y)
    X = clip_outliers(X, low_pct=1, high_pct=99)

    return X, y, feature_cols


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ============================================================================
# SPLIT & SCALE
# ============================================================================

def split_and_scale(X, y, config: Config, seed: int):
    ds = Dataset(X, y)

    ds_trainval, ds_test = ds.train_test_split(
        test_size=config.TEST_SIZE, stratify=True, seed=seed
    )
    val_frac = config.VALIDATION_SIZE / (1.0 - config.TEST_SIZE)
    ds_train, ds_val = ds_trainval.train_test_split(
        test_size=val_frac, stratify=True, seed=seed
    )

    X_train, y_train = ds_train.X, ds_train.y
    X_val,   y_val   = ds_val.X,   ds_val.y
    X_test,  y_test  = ds_test.X,  ds_test.y

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    return {
        "X_train": X_train_s, "y_train": y_train,
        "X_val":   X_val_s,   "y_val":   y_val,
        "X_test":  X_test_s,  "y_test":  y_test,
        "scaler":  scaler,
        "n_train": len(X_train),
        "n_val":   len(X_val),
        "n_test":  len(X_test),
    }


# ============================================================================
# TRAINING
# ============================================================================

def train(splits: dict, config: Config, seed: int):
    model = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        min_samples_split=config.MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.MIN_SAMPLES_LEAF,
        max_features=config.MAX_FEATURES,
        class_weight="balanced",
        random_state=seed,
    )
    t0 = time.time()
    model.fit(splits["X_train"], splits["y_train"])
    elapsed = time.time() - t0
    return model, elapsed


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, X, y, split_name: str, feature_names=None, verbose=True):
    t0 = time.time()
    y_pred  = model.predict(X)
    y_score = model.predict_proba(X)
    inf_time = time.time() - t0

    throughput = len(X) / inf_time if inf_time > 0 else float("inf")

    acc   = accuracy(y, y_pred)
    prec  = precision(y, y_pred)
    rec   = recall(y, y_pred)
    f1    = f1_score(y, y_pred)
    auc   = roc_auc(y, y_score)
    cm, _ = confusion_matrix(y, y_pred)

    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    if verbose:
        print(f"\n  Inference: {inf_time:.4f}s  |  {throughput:,.0f} samples/sec")
        print(f"\n  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"  ROC-AUC   : {auc:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    +----------------+------------+------------+")
        print(f"    |                | Pred Benign| Pred Attack|")
        print(f"    +----------------+------------+------------+")
        print(f"    | Actual Benign  | {tn:>10,} | {fp:>10,} |")
        print(f"    | Actual Attack  | {fn:>10,} | {tp:>10,} |")
        print(f"    +----------------+------------+------------+")
        print(f"\n  FPR : {fpr:.6f}  ({fpr*100:.4f}%)")
        print(f"  FNR : {fnr:.6f}  ({fnr*100:.4f}%)")

        if feature_names and split_name.lower() == "test":
            try:
                importances = model.feature_importances_
                ranked = sorted(
                    zip(feature_names, importances),
                    key=lambda x: x[1], reverse=True
                )
                print(f"\n  Top 10 feature importances:")
                for name, imp in ranked[:10]:
                    print(f"    {name:<42} {imp:.6f}")
            except AttributeError:
                pass

    # Collect feature importances for JSON storage
    feat_imp = {}
    if feature_names:
        try:
            feat_imp = dict(zip(feature_names, model.feature_importances_))
        except AttributeError:
            pass

    return {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "auc": auc, "fpr": fpr, "fnr": fnr,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "inference_time": inf_time, "throughput": throughput,
        "feature_importances": feat_imp,
    }


# ============================================================================
# STATISTICS HELPERS
# ============================================================================

def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

def _std(vals):
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

def _aggregate(run_metrics: list, key: str):
    vals = [r[key] for r in run_metrics]
    return {"mean": _mean(vals), "std": _std(vals), "runs": vals}

def _aggregate_importances(run_metrics: list):
    """Average feature importances across repeats."""
    all_imps = [r["feature_importances"] for r in run_metrics if r["feature_importances"]]
    if not all_imps:
        return {}
    features = list(all_imps[0].keys())
    return {
        f: {"mean": _mean([d[f] for d in all_imps]),
            "std":  _std([d[f] for d in all_imps])}
        for f in features
    }


# ============================================================================
# SAVE ARTIFACTS
# ============================================================================

def save_artifacts(model, scaler, config: Config, suffix: str = ""):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path  = config.MODEL_DIR / f"iotids_rf_model{suffix}.pkl"
    scaler_path = config.MODEL_DIR / f"iotids_rf_scaler{suffix}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n  Model  saved: {model_path}")
    print(f"  Scaler saved: {scaler_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="iotids Random Forest — CIC-IDS-2017")
    parser.add_argument("--data",    default=None,  help="Override CSV path")
    parser.add_argument("--sample",  default=None,  type=float,
                        help="Fraction for stratified sample (e.g. 0.1)")
    parser.add_argument("--repeats", default=1,     type=int,
                        help="Number of independent sample+train+eval runs (default 1)")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  RANDOM FOREST BASELINE  —  iotids Library".center(80))
    print("  No TensorFlow · No scikit-learn · No pandas".center(80))
    print("=" * 80)

    config = Config()
    if args.data:
        config.DATA_FILE = Path(args.data)
    if args.sample is not None:
        config.USE_SAMPLE  = True
        config.SAMPLE_FRAC = args.sample
    config.REPEATS = args.repeats

    # ── Load full dataset once ────────────────────────────────────────────
    raw_data = load_data(config)

    # Storage for all repeat results
    all_runs   = []          # one entry per repeat, stored in JSON
    test_metrics_runs  = []
    val_metrics_runs   = []
    train_metrics_runs = []
    train_times        = []

    for repeat in range(config.REPEATS):
        repeat_seed = RANDOM_SEED + repeat   # different seed each repeat
        print("\n" + "=" * 80)
        print(f"  REPEAT {repeat + 1} / {config.REPEATS}  (seed={repeat_seed})".center(80))
        print("=" * 80)

        # ── Sample ───────────────────────────────────────────────────────
        if config.USE_SAMPLE and config.SAMPLE_FRAC < 1.0:
            print(f"\n  Stratified {config.SAMPLE_FRAC*100:.0f}% sample (seed={repeat_seed})...")
            data = _stratified_sample(raw_data, config.SAMPLE_FRAC, repeat_seed)
            n_rows = len(next(iter(data.values())))
            print(f"  Sample size: {n_rows:,}")
        else:
            data = raw_data

        # ── Preprocess ───────────────────────────────────────────────────
        print("\n--- Preprocessing ---")
        X, y, feat_names = preprocess(data, config)
        n_samples = len(X)
        n_attack  = sum(y)
        n_benign  = n_samples - n_attack
        print(f"  Samples: {n_samples:,}  |  Benign: {n_benign:,}  |  Attack: {n_attack:,}")

        # ── Split & scale ────────────────────────────────────────────────
        splits = split_and_scale(X, y, config, seed=repeat_seed)
        print(f"  Train: {splits['n_train']:,}  Val: {splits['n_val']:,}  Test: {splits['n_test']:,}")

        # ── Train ────────────────────────────────────────────────────────
        print(f"\n--- Training (n_estimators={config.N_ESTIMATORS}, "
              f"max_depth={config.MAX_DEPTH}) ---")
        model, train_time = train(splits, config, seed=repeat_seed)
        print(f"  Done in {train_time:.2f}s")
        train_times.append(train_time)

        # ── Evaluate ─────────────────────────────────────────────────────
        verbose = (config.REPEATS == 1)   # full printout only for single runs

        print("\n--- Evaluating TRAIN ---")
        m_train = evaluate(model, splits["X_train"], splits["y_train"],
                           "Train", verbose=verbose)

        print("\n--- Evaluating VALIDATION ---")
        m_val = evaluate(model, splits["X_val"], splits["y_val"],
                         "Validation", verbose=verbose)

        print("\n--- Evaluating TEST ---")
        m_test = evaluate(model, splits["X_test"], splits["y_test"],
                          "Test", feat_names, verbose=verbose)

        # Always print a compact per-repeat summary
        print(f"\n  [Repeat {repeat+1}] "
              f"Acc={m_test['accuracy']*100:.2f}%  "
              f"Prec={m_test['precision']:.4f}  "
              f"Rec={m_test['recall']:.4f}  "
              f"F1={m_test['f1']:.4f}  "
              f"AUC={m_test['auc']:.4f}  "
              f"FPR={m_test['fpr']:.4f}  "
              f"FNR={m_test['fnr']:.4f}  "
              f"Time={train_time:.1f}s")

        train_metrics_runs.append(m_train)
        val_metrics_runs.append(m_val)
        test_metrics_runs.append(m_test)

        # ── Save model for this repeat ────────────────────────────────────
        suffix = f"_repeat{repeat+1}" if config.REPEATS > 1 else ""
        save_artifacts(model, splits["scaler"], config, suffix=suffix)

        # ── Per-repeat JSON record ────────────────────────────────────────
        all_runs.append({
            "repeat":      repeat + 1,
            "seed":        repeat_seed,
            "sample_frac": config.SAMPLE_FRAC if config.USE_SAMPLE else 1.0,
            "n_train":     splits["n_train"],
            "n_val":       splits["n_val"],
            "n_test":      splits["n_test"],
            "train_time_s": round(train_time, 4),
            "train": {k: round(m_train[k], 6)
                      for k in ("accuracy","precision","recall","f1","auc","fpr","fnr")},
            "val":   {k: round(m_val[k], 6)
                      for k in ("accuracy","precision","recall","f1","auc","fpr","fnr")},
            "test":  {k: round(m_test[k], 6)
                      for k in ("accuracy","precision","recall","f1","auc","fpr","fnr")},
            "feature_importances": {
                k: round(v, 6) for k, v in m_test["feature_importances"].items()
            },
        })

    # ============================================================
    # AGGREGATE SUMMARY (only meaningful when repeats > 1)
    # ============================================================
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY".center(80))
    print("=" * 80)

    metrics_to_show = ["accuracy", "precision", "recall", "f1", "auc", "fpr", "fnr"]

    if config.REPEATS > 1:
        print(f"\n  Repeats : {config.REPEATS}  |  "
              f"Sample  : {config.SAMPLE_FRAC*100:.0f}%  |  "
              f"Samples/run ~{test_metrics_runs[0].get('throughput', 0)*0:.0f}")
        print(f"\n  {'Metric':<14} {'Mean':>10} {'±Std':>10}  {'Runs'}")
        print(f"  {'-'*60}")
        for m in metrics_to_show:
            agg = _aggregate(test_metrics_runs, m)
            runs_str = "  ".join(f"{v:.4f}" for v in agg["runs"])
            print(f"  {m:<14} {agg['mean']:>10.4f} {agg['std']:>10.4f}  [{runs_str}]")
        avg_time = _mean(train_times)
        std_time = _std(train_times)
        print(f"\n  {'Train time':<14} {avg_time:>10.2f}s {std_time:>9.2f}s")
    else:
        m = test_metrics_runs[0]
        print(f"\n  Test Accuracy  : {m['accuracy']:.4f}  ({m['accuracy']*100:.2f}%)")
        print(f"  Test Precision : {m['precision']:.4f}")
        print(f"  Test Recall    : {m['recall']:.4f}")
        print(f"  Test F1-Score  : {m['f1']:.4f}")
        print(f"  Test ROC-AUC   : {m['auc']:.4f}")
        print(f"  FPR            : {m['fpr']:.6f}  ({m['fpr']*100:.4f}%)")
        print(f"  FNR            : {m['fnr']:.6f}  ({m['fnr']*100:.4f}%)")
        print(f"  Training time  : {train_times[0]:.2f}s")

    print(f"\n  Varun (2025) baseline : 99.2%")
    best_acc = max(r["accuracy"] for r in test_metrics_runs)
    print(f"  iotids best result    : {best_acc*100:.2f}%")

    status = "PASS — ready for federated learning" \
             if _mean([r["accuracy"] for r in test_metrics_runs]) >= 0.95 \
             else "WARN — below 95%, consider tuning"
    print(f"\n  Status: {status}")

    # ============================================================
    # SAVE JSON
    # ============================================================
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.OUTPUT_DIR / "rf_results.json"

    # Aggregate importances across repeats
    agg_importances = _aggregate_importances(test_metrics_runs)
    top_features = sorted(agg_importances.items(),
                          key=lambda x: x[1]["mean"], reverse=True)

    output = {
        "meta": {
            "dataset":          str(config.DATA_FILE),
            "sample_frac":      config.SAMPLE_FRAC if config.USE_SAMPLE else 1.0,
            "repeats":          config.REPEATS,
            "n_estimators":     config.N_ESTIMATORS,
            "max_depth":        config.MAX_DEPTH,
            "min_samples_split":config.MIN_SAMPLES_SPLIT,
            "min_samples_leaf": config.MIN_SAMPLES_LEAF,
            "max_features":     config.MAX_FEATURES,
            "baseline_ref":     "Varun (2025) 99.2%",
        },
        "aggregate": {
            m: _aggregate(test_metrics_runs, m) for m in metrics_to_show
        } if config.REPEATS > 1 else {},
        "feature_importances_aggregate": {
            f: {"mean": round(v["mean"], 6), "std": round(v["std"], 6)}
            for f, v in top_features
        },
        "runs": all_runs,
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved: {json_path}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()