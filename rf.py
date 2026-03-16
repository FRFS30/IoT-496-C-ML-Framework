"""
CIC-IDS-2017 Random Forest Baseline — iotids Library
=====================================================
Trains and evaluates a Random Forest classifier using only the iotids
Python library. No TensorFlow, scikit-learn, or pandas dependency.

Expected Result: ~99% accuracy on the CIC-IDS-2017 test set.

Usage:
    python random_forest.py
    python random_forest.py --data path/to/clean_dataset.csv
    python random_forest.py --sample 0.2   # 20% stratified sample for quick runs
"""

import argparse
import time
from pathlib import Path

# ── iotids imports only ───────────────────────────────────────────────────────
from python.iotids.data.csv_reader   import read_csv
from python.iotids.data.preprocessing import (
    RobustScaler,
    LabelEncoder,
    clip_outliers,
    replace_inf,
    drop_nan_rows,
)
from python.iotids.data.dataset       import Dataset
from python.iotids.forest.random_forest import RandomForestClassifier
from python.iotids.forest.serializer  import save_rf, load_rf
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
    DATA_FILE      = Path("processed_reduced/clean_dataset_new.csv")
    OUTPUT_DIR     = Path("results/iotids_rf")
    MODEL_DIR      = Path("models/iotids_rf")

    # Dataset
    USE_SAMPLE     = False
    SAMPLE_FRAC    = 0.20

    # Model hyperparameters — mirrors the optimised sklearn baseline
    N_ESTIMATORS       = 100
    MAX_DEPTH          = 20
    MIN_SAMPLES_SPLIT  = 50
    MIN_SAMPLES_LEAF   = 20
    MAX_FEATURES       = "sqrt"   # sqrt(n_features) per split

    # Split sizes
    TEST_SIZE       = 0.20
    VALIDATION_SIZE = 0.10

    # Non-feature columns in CIC-IDS-2017
    EXCLUDE_COLS = {
        "Label", "Flow_ID", "Source_IP", "Destination_IP",
        "Timestamp", "Source_Port", "Destination_Port",
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(config: Config):
    """Load CIC-IDS-2017 via iotids csv_reader."""
    print("\n" + "=" * 80)
    print("LOADING CIC-IDS-2017 DATASET")
    print("=" * 80)

    if not config.DATA_FILE.exists():
        raise FileNotFoundError(f"\n  Dataset not found: {config.DATA_FILE}")

    print(f"\n  Loading: {config.DATA_FILE}")
    data = read_csv(str(config.DATA_FILE))          # dict[col -> list]

    n_rows = len(next(iter(data.values())))
    print(f"  Loaded {n_rows:,} rows, {len(data)} columns")

    # ── Class distribution ────────────────────────────────────────────────
    labels_raw = data["Label"]
    label_counts: dict = {}
    for lbl in labels_raw:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    print(f"\n  Class distribution:")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {str(lbl):<35} {cnt:>10,}  ({cnt / n_rows * 100:>5.2f}%)")

    # ── Optional stratified sample ────────────────────────────────────────
    if config.USE_SAMPLE and config.SAMPLE_FRAC < 1.0:
        print(f"\n  Stratified {config.SAMPLE_FRAC*100:.0f}% sample...")
        data = _stratified_sample(data, config.SAMPLE_FRAC, RANDOM_SEED)
        n_rows = len(next(iter(data.values())))
        print(f"  Sample size: {n_rows:,}")

    return data


def _stratified_sample(data: dict, frac: float, seed: int) -> dict:
    """Return a stratified random fraction of the raw column dict."""
    import random as _rng
    _rng.seed(seed)

    labels = data["Label"]
    # Group indices by class
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
    """
    Clean data and build feature matrix X and binary label vector y.

    Pipeline (mirrors the standalone baseline):
      1. Strip Inf / -Inf  ->  NaN
      2. Drop all-NaN rows
      3. Clip outliers to [1st, 99th] percentile
      4. Build binary label  (0 = BENIGN, 1 = ATTACK)
      5. Select all numeric feature columns
    """
    print("\n" + "=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)

    # ── Identify numeric feature columns ─────────────────────────────────
    feature_cols = [
        col for col in data
        if col not in config.EXCLUDE_COLS
        and all(
            isinstance(v, (int, float)) or
            (isinstance(v, str) and _is_numeric(v))
            for v in data[col][:100]          # quick type probe on first 100
        )
    ]

    print(f"\n  Feature columns: {len(feature_cols)}")

    n_rows = len(next(iter(data.values())))

    # ── Build float X matrix (list of lists) ─────────────────────────────
    print("  Converting to float arrays...")
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

    # Transpose to row-major: X[i] = feature vector for sample i
    X = [[X_cols[j][i] for j in range(len(feature_cols))]
         for i in range(n_rows)]

    # ── Binary label ──────────────────────────────────────────────────────
    benign = {"BENIGN", "Benign", "benign"}
    raw_labels = data["Label"]
    y = []
    for lbl in raw_labels:
        try:
            y.append(int(lbl))           # already encoded 0/1
        except (ValueError, TypeError):
            y.append(0 if str(lbl) in benign else 1)

    # ── Replace Inf / NaN ─────────────────────────────────────────────────
    print("  Replacing Inf values...")
    X = replace_inf(X)

    print("  Dropping NaN rows...")
    X, y = drop_nan_rows(X, y)

    # ── Clip outliers ─────────────────────────────────────────────────────
    print("  Clipping outliers [1st, 99th pct]...")
    X = clip_outliers(X, low_pct=1, high_pct=99)

    n_samples = len(X)
    n_feat    = len(X[0]) if X else 0
    n_attack  = sum(y)
    n_benign  = n_samples - n_attack

    print(f"\n  After cleaning:")
    print(f"    Samples  : {n_samples:,}")
    print(f"    Features : {n_feat}")
    print(f"    Benign   : {n_benign:,}  ({n_benign/n_samples*100:.1f}%)")
    print(f"    Attack   : {n_attack:,}  ({n_attack/n_samples*100:.1f}%)")

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

def split_and_scale(X, y, config: Config):
    """Stratified train / val / test split + RobustScaler fit on train only."""
    print("\n" + "=" * 80)
    print("DATA SPLITTING AND SCALING")
    print("=" * 80)

    ds = Dataset(X, y)

    # First cut: (train+val) vs test
    ds_trainval, ds_test = ds.train_test_split(
        test_size=config.TEST_SIZE, stratify=True, seed=RANDOM_SEED
    )

    # Second cut: train vs val
    val_frac = config.VALIDATION_SIZE / (1.0 - config.TEST_SIZE)
    ds_train, ds_val = ds_trainval.train_test_split(
        test_size=val_frac, stratify=True, seed=RANDOM_SEED
    )

    X_train, y_train = ds_train.X, ds_train.y
    X_val,   y_val   = ds_val.X,   ds_val.y
    X_test,  y_test  = ds_test.X,  ds_test.y

    n_total = len(X)
    print(f"\n  Train : {len(X_train):,}  ({len(X_train)/n_total*100:.1f}%)")
    print(f"  Val   : {len(X_val):,}  ({len(X_val)/n_total*100:.1f}%)")
    print(f"  Test  : {len(X_test):,}  ({len(X_test)/n_total*100:.1f}%)")

    print("\n  Fitting RobustScaler on training data...")
    scaler = RobustScaler(quantile_range=(5, 95))
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    print("  Scaling complete.")

    return {
        "X_train": X_train_s, "y_train": y_train,
        "X_val":   X_val_s,   "y_val":   y_val,
        "X_test":  X_test_s,  "y_test":  y_test,
        "scaler":  scaler,
    }


# ============================================================================
# TRAINING
# ============================================================================

def train(splits: dict, config: Config):
    """Fit RandomForestClassifier from iotids.forest."""
    print("\n" + "=" * 80)
    print("TRAINING RANDOM FOREST  (iotids.forest)")
    print("=" * 80)

    print(f"\n  n_estimators    : {config.N_ESTIMATORS}")
    print(f"  max_depth       : {config.MAX_DEPTH}")
    print(f"  min_samples_split: {config.MIN_SAMPLES_SPLIT}")
    print(f"  min_samples_leaf : {config.MIN_SAMPLES_LEAF}")
    print(f"  max_features    : {config.MAX_FEATURES}")

    model = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        min_samples_split=config.MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.MIN_SAMPLES_LEAF,
        max_features=config.MAX_FEATURES,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    print(f"\n  Training on {len(splits['X_train']):,} samples...")
    t0 = time.time()
    model.fit(splits["X_train"], splits["y_train"])
    elapsed = time.time() - t0

    print(f"  Done in {elapsed:.2f}s  ({elapsed/60:.2f} min)")
    return model, elapsed


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, X, y, split_name: str, feature_names=None):
    """Compute and print all IDS metrics using iotids.metrics."""
    print("\n" + "=" * 80)
    print(f"EVALUATING — {split_name.upper()}")
    print("=" * 80)

    t0 = time.time()
    y_pred  = model.predict(X)
    y_score = model.predict_proba(X)        # probability for class 1
    inf_time = time.time() - t0

    throughput = len(X) / inf_time if inf_time > 0 else float("inf")
    print(f"\n  Inference: {inf_time:.4f}s  |  {throughput:,.0f} samples/sec")

    acc   = accuracy(y, y_pred)
    prec  = precision(y, y_pred)
    rec   = recall(y, y_pred)
    f1    = f1_score(y, y_pred)
    auc   = roc_auc(y, y_score)
    cm, _ = confusion_matrix(y, y_pred)

    
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

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

    # Feature importances (test set only, if available)
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

    return {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "auc": auc, "fpr": fpr, "fnr": fnr,
        "confusion_matrix": cm,
        "inference_time": inf_time, "throughput": throughput,
    }


# ============================================================================
# SAVE ARTIFACTS
# ============================================================================

def save_artifacts(model, scaler, config: Config):
    """Persist model and scaler using iotids serialisers."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path  = config.MODEL_DIR / "iotids_rf_model.bin"
    scaler_path = config.MODEL_DIR / "iotids_rf_scaler.bin"

    save_rf(model, str(model_path))
    save(scaler, str(scaler_path))

    print(f"\n  Model  saved: {model_path}")
    print(f"  Scaler saved: {scaler_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="iotids Random Forest — CIC-IDS-2017")
    parser.add_argument("--data",   default=None,  help="Override CSV path")
    parser.add_argument("--sample", default=None,  type=float,
                        help="Fraction for quick stratified sample (e.g. 0.2)")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    print("\n" + "=" * 80)
    print("  RANDOM FOREST BASELINE  —  iotids Library".center(80))
    print("  No TensorFlow · No scikit-learn · No pandas".center(80))
    print("=" * 80)

    config = Config()
    if args.data:
        config.DATA_FILE = Path(args.data)
    if args.sample is not None:
        config.USE_SAMPLE   = True
        config.SAMPLE_FRAC  = args.sample

    # ── Pipeline ──────────────────────────────────────────────────────────
    data               = load_data(config)
    X, y, feat_names   = preprocess(data, config)
    splits             = split_and_scale(X, y, config)
    model, train_time  = train(splits, config)

    m_train = evaluate(model, splits["X_train"], splits["y_train"], "Train")
    m_val   = evaluate(model, splits["X_val"],   splits["y_val"],   "Validation")
    m_test  = evaluate(model, splits["X_test"],  splits["y_test"],  "Test", feat_names)

    save_artifacts(model, splits["scaler"], config)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY".center(80))
    print("=" * 80)
    print(f"\n  Test Accuracy  : {m_test['accuracy']:.4f}  ({m_test['accuracy']*100:.2f}%)")
    print(f"  Test Precision : {m_test['precision']:.4f}")
    print(f"  Test Recall    : {m_test['recall']:.4f}")
    print(f"  Test F1-Score  : {m_test['f1']:.4f}")
    print(f"  Test ROC-AUC   : {m_test['auc']:.4f}")
    print(f"  FPR            : {m_test['fpr']:.6f}  ({m_test['fpr']*100:.4f}%)")
    print(f"  FNR            : {m_test['fnr']:.6f}  ({m_test['fnr']*100:.4f}%)")
    print(f"  Training time  : {train_time:.2f}s")
    print(f"\n  Varun (2025) baseline : 99.2%")
    print(f"  iotids result         : {m_test['accuracy']*100:.2f}%")

    status = "PASS — ready for federated learning" if m_test["accuracy"] >= 0.95 \
             else "WARN — below 95%, consider tuning"
    print(f"\n  Status: {status}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()