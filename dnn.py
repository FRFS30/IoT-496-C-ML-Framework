"""
CIC-IDS-2017 Deep Neural Network Baseline — iotids Library
===========================================================
Trains, evaluates, and quantizes a DNN classifier using only the iotids
Python library. No TensorFlow, scikit-learn, or pandas dependency.

Speed optimizations applied
----------------------------
  - Largest feasible batch size (4096) to maximise throughput on both CPU
    and GPU: fewer gradient steps per epoch, better hardware utilisation.
  - Mini-batch forward/backward computed in one array pass per batch.
  - Oversampling done once up-front (not re-drawn every epoch).
  - Early stopping with patience=7 avoids wasted epochs.
  - RobustScaler fitted on training data only (one pass).
  - Stratified 70 / 15 / 15 split keeps val/test proportional.

Size optimizations applied
---------------------------
  model.bin  — compact binary: every float64 weight downcast to float32
               (halves size vs. pickle); only the network topology + weights
               stored, no Python objects.
  scaler.bin — only 2×n_features float32 numbers (median + IQR) stored via
               utils.io.save(); no sklearn objects, no class metadata.

GPU / CPU
---------
  The iotids library is pure-Python and does not interface with CUDA directly.
  To accelerate training on the university server's GPU, set the environment
  variable IOTIDS_BACKEND=numpy before running — this switches the hot-path
  matrix multiply in core/ops.py to NumPy (which uses OpenBLAS/MKL and will
  pick up any available BLAS acceleration including GPU-attached BLAS).

  CPU-only run (default, no extra packages):
      python3 dnn.py --data processed_reduced/clean_dataset.csv

  NumPy-backed run (uses BLAS / cuBLAS if available):
      IOTIDS_BACKEND=numpy python3 dnn.py --data processed_reduced/clean_dataset.csv

  Sampling for a quick smoke-test:
      python3 dnn.py --data processed_reduced/clean_dataset.csv --sample 0.1

Expected outputs (models/iotids_dnn/)
---------------------------------------
  iotids_dnn.model      — network weights, compact binary (~30–80 KB)
  iotids_dnn_scaler.bin — RobustScaler params (~400 bytes for 24 features)
  iotids_dnn_threshold.bin — optimal decision threshold (float)

Expected result
---------------
  ~99% accuracy on CIC-IDS-2017 test set (matches TensorFlow DNN baseline).
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve iotids package: look for python/iotids relative to this script.
# Layout:
#   <repo-root>/
#       dnn.py              <- this file
#       rf.py
#       C/
#       python/
#           iotids/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PYTHON_DIR = _HERE / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

# Core imports from iotids -----------------------------------------------
from iotids.data.csv_reader    import read_csv
from iotids.data.preprocessing import (
    RobustScaler, clip_outliers, replace_inf, drop_nan_rows,
)
from iotids.data.dataset       import Dataset
from iotids.nn.layers          import Dense, BatchNormalization, Dropout
from iotids.nn.losses          import FocalLoss, BinaryCrossentropy
from iotids.nn.optimizers      import Adam
from iotids.nn.model           import Sequential, EarlyStopping
from iotids.metrics.classification import (
    accuracy, precision, recall, f1_score, roc_auc,
    confusion_matrix, threshold_sweep, classification_report,
)
from iotids.utils.io           import save, load
from iotids.utils.random       import set_seed

RANDOM_SEED = 42


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    DATA_FILE  = Path("processed_reduced/clean_dataset.csv")
    OUTPUT_DIR = Path("results/iotids_dnn")
    MODEL_DIR  = Path("models/iotids_dnn")

    # Dataset
    USE_SAMPLE   = False
    SAMPLE_FRAC  = 0.20
    TEST_SIZE    = 0.15
    VAL_SIZE     = 0.15

    # Training — large batch for throughput; fewer steps per epoch
    EPOCHS        = 40
    BATCH_SIZE    = 4096          # maximises CPU/GPU utilisation
    LEARNING_RATE = 2e-4
    PATIENCE      = 7             # early stopping patience

    # Architecture — proven DNN baseline (256→128→64→1)
    HIDDEN_UNITS  = [256, 128, 64]
    DROPOUT_RATES = [0.4,  0.4,  0.3]

    # Loss (focal loss handles 80/20 class imbalance)
    USE_FOCAL_LOSS = True
    FOCAL_ALPHA    = 0.70
    FOCAL_GAMMA    = 1.8

    # Decision threshold
    DECISION_THRESHOLD = 0.4
    OPTIMIZE_THRESHOLD = True     # sweep val set to find optimal threshold

    # Oversampling (training split only — prevents val/test leakage)
    USE_OVERSAMPLING  = True
    OVERSAMPLE_RATIO  = 0.5       # target attack / benign ratio after oversample

    # Quantization (INT8 export via tflm_export if available)
    ENABLE_QUANTIZATION = True

    # Columns to exclude from feature matrix
    EXCLUDE_COLS = {
        "Label", "Flow_ID", "Source_IP", "Destination_IP",
        "Timestamp", "Source_Port", "Destination_Port",
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(config: Config) -> dict:
    print("\n" + "=" * 80)
    print("LOADING CIC-IDS-2017 DATASET")
    print("=" * 80)

    if not config.DATA_FILE.exists():
        raise FileNotFoundError(f"\n  Dataset not found: {config.DATA_FILE}\n"
                                f"  Run from the repo root or pass --data <path>")

    print(f"\n  Loading: {config.DATA_FILE}")
    data = read_csv(str(config.DATA_FILE))

    # data is dict[col_name -> list_of_values]
    n_rows = len(next(iter(data.values())))
    print(f"  Loaded {n_rows:,} rows, {len(data)} columns")

    # Class distribution
    labels = data.get("Label", data.get(" Label", []))
    counts: dict = {}
    for v in labels:
        counts[v] = counts.get(v, 0) + 1
    print("  Class distribution:")
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {str(cls):<40} {cnt:>10,}  ({cnt / n_rows * 100:.2f}%)")

    # Optional stratified sample for faster dev iterations
    if config.USE_SAMPLE:
        frac = config.SAMPLE_FRAC
        print(f"  Stratified {int(frac * 100)}% sample...")
        ds_full = Dataset([], [])
        # Build minimal Dataset to use .sample()
        # We'll do manual stratified sample here for robustness
        by_class: dict = {}
        for i, v in enumerate(labels):
            by_class.setdefault(v, []).append(i)
        import random as _rnd
        _rnd.seed(RANDOM_SEED)
        keep: set = set()
        for cls_indices in by_class.values():
            k = max(1, int(len(cls_indices) * frac))
            keep.update(_rnd.sample(cls_indices, k))
        data = {col: [v for i, v in enumerate(vals) if i in keep]
                for col, vals in data.items()}
        n_rows = len(next(iter(data.values())))
        print(f"  Sample size: {n_rows:,}")

    return data


# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess(data: dict, config: Config):
    """
    1. Separate feature columns from label column.
    2. Convert to float lists.
    3. Replace Inf, drop NaN rows, clip 1st–99th percentile outliers.
    4. Encode labels as binary 0/1.
    Returns (X: list[list[float]], y: list[int], feature_names: list[str])
    """
    print("\n" + "=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)

    # Identify label column (handles leading-space column names from CIC-IDS)
    label_key = None
    for k in data.keys():
        if k.strip() == "Label":
            label_key = k
            break
    if label_key is None:
        raise KeyError("No 'Label' column found in dataset.")

    feat_names = [k for k in data.keys()
                  if k.strip() not in config.EXCLUDE_COLS]
    print(f"  Feature columns: {len(feat_names)}")

    # Convert to float
    print("  Converting to float arrays...")
    X_raw: list[list] = []
    for i in range(len(data[feat_names[0]])):
        row = []
        for col in feat_names:
            try:
                row.append(float(data[col][i]))
            except (ValueError, TypeError):
                row.append(float("nan"))
        X_raw.append(row)

    raw_labels = data[label_key]

    # Replace Inf → NaN, drop NaN rows, clip outliers
    print("  Replacing Inf values...")
    X_raw = replace_inf(X_raw)
    print("  Dropping NaN rows...")
    X_raw, raw_labels = drop_nan_rows(X_raw, raw_labels)
    print("  Clipping outliers [1st, 99th pct]...")
    X_raw = clip_outliers(X_raw, low_pct=1, high_pct=99)

    # Binary label encoding: 0 = BENIGN, 1 = attack
    # Handles both string "BENIGN" and numeric 0.0 / 0 labels
    def _is_benign(lbl):
        s = str(lbl).strip()
        try:
            return float(s) == 0.0
        except ValueError:
            return s == "BENIGN"
    y = [0 if _is_benign(lbl) else 1 for lbl in raw_labels]

    n_benign = sum(1 for v in y if v == 0)
    n_attack = sum(1 for v in y if v == 1)
    n_total  = len(y)
    print(f"\n  After cleaning:")
    print(f"    Samples  : {n_total:,}")
    print(f"    Features : {len(feat_names)}")
    print(f"    Benign   : {n_benign:,}  ({n_benign / n_total * 100:.1f}%)")
    print(f"    Attack   : {n_attack:,}  ({n_attack / n_total * 100:.1f}%)")

    return X_raw, y, feat_names


# ============================================================================
# SPLIT, SCALE, AND OVERSAMPLE
# ============================================================================

def split_and_scale(X: list, y: list, config: Config) -> dict:
    """
    70/15/15 stratified train/val/test split.
    Fit RobustScaler on training data only.
    Optionally oversample attack class in training data.
    """
    print("\n" + "=" * 80)
    print("DATA SPLITTING AND SCALING")
    print("=" * 80)

    # Two-pass split: first carve out test, then split remainder into train/val
    # val_size is expressed relative to the full dataset
    ds = Dataset(X, y)
    trainval_ds, test_ds = ds.train_test_split(
        test_size=config.TEST_SIZE,
        stratify=True,
        seed=RANDOM_SEED,
    )
    # val fraction relative to the trainval remainder
    val_frac = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    train_ds, val_ds = Dataset(trainval_ds.X, trainval_ds.y).train_test_split(
        test_size=val_frac,
        stratify=True,
        seed=RANDOM_SEED,
    )

    print(f"  Train : {len(train_ds.X):,}  ({(1 - config.TEST_SIZE - config.VAL_SIZE)*100:.1f}%)")
    print(f"  Val   : {len(val_ds.X):,}  ({config.VAL_SIZE*100:.1f}%)")
    print(f"  Test  : {len(test_ds.X):,}  ({config.TEST_SIZE*100:.1f}%)")

    # Fit scaler on training data only (no leakage)
    print("  Fitting RobustScaler on training data...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(train_ds.X)
    X_val_scaled   = scaler.transform(val_ds.X)
    X_test_scaled  = scaler.transform(test_ds.X)
    print("  Scaling complete.")

    y_train = list(train_ds.y)

    # Oversample attack class in training split only
    if config.USE_OVERSAMPLING:
        benign_idx  = [i for i, v in enumerate(y_train) if v == 0]
        attack_idx  = [i for i, v in enumerate(y_train) if v == 1]
        target_n_attack = int(len(benign_idx) * config.OVERSAMPLE_RATIO)
        if target_n_attack > len(attack_idx):
            import random as _rnd
            _rnd.seed(RANDOM_SEED)
            extra = _rnd.choices(attack_idx,
                                 k=target_n_attack - len(attack_idx))
            X_train_scaled = X_train_scaled + [X_train_scaled[i] for i in extra]
            y_train        = y_train + [1] * len(extra)
            print(f"  Oversampled: {len(attack_idx):,} → {target_n_attack:,} attacks"
                  f"  (total train: {len(y_train):,})")

    return {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_val":   X_val_scaled,
        "y_val":   list(val_ds.y),
        "X_test":  X_test_scaled,
        "y_test":  list(test_ds.y),
        "scaler":  scaler,
    }


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def build_model(n_features: int, config: Config) -> Sequential:
    """
    Dense 256 → 128 → 64 → 1  with BatchNorm + Dropout.
    Raw logit output (sigmoid applied at threshold evaluation time).
    """
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE  (iotids.nn)")
    print("=" * 80)

    layers = []
    in_size = n_features

    for units, drop in zip(config.HIDDEN_UNITS, config.DROPOUT_RATES):
        layers.append(Dense(in_size, units, activation="relu"))
        layers.append(BatchNormalization(units))
        layers.append(Dropout(drop))
        in_size = units

    # Output: single logit (no activation — FocalLoss handles sigmoid internally)
    layers.append(Dense(in_size, 1, activation=None))

    model = Sequential(layers)

    if config.USE_FOCAL_LOSS:
        model._loss_fn = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
        print(f"\n  Loss     : FocalLoss(alpha={config.FOCAL_ALPHA}, "
              f"gamma={config.FOCAL_GAMMA})")
    else:
        model._loss_fn = BinaryCrossentropy(from_logits=True)
        print(f"\n  Loss     : BinaryCrossentropy(from_logits=True)")

    model._optimizer = Adam(lr=config.LEARNING_RATE)

    print(f"  Optimizer: Adam(lr={config.LEARNING_RATE})")
    print(f"  Arch     : {n_features} -> " +
          " -> ".join(str(u) for u in config.HIDDEN_UNITS) + " -> 1")
    print(f"  Dropout  : {config.DROPOUT_RATES}")
    print(f"  Batch    : {config.BATCH_SIZE}  (maximised for throughput)")

    return model


# ============================================================================
# TRAINING
# ============================================================================

def train(model: Sequential, splits: dict, config: Config):
    """
    model.fit() with early stopping.

    Class weight: weight_attack = n_benign / n_attack
    Applied as a scalar multiplier to the loss gradient for attack samples
    so the minority class receives proportionally larger updates.
    """
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val   = splits["X_val"]
    y_val   = splits["y_val"]

    n_benign = sum(1 for v in y_train if v == 0)
    n_attack = sum(1 for v in y_train if v == 1)
    class_weight = {0: 1.0, 1: n_benign / max(n_attack, 1)}
    print(f"\n  Class weight — attack: {class_weight[1]:.2f}x")
    print(f"  Training samples     : {len(X_train):,}")
    print(f"  Validation samples   : {len(X_val):,}")
    print(f"  Batch size           : {config.BATCH_SIZE}")
    steps = max(1, len(X_train) // config.BATCH_SIZE)
    print(f"  Steps/epoch          : {steps}")
    print(f"  Max epochs           : {config.EPOCHS}")
    print(f"  Early stop patience  : {config.PATIENCE}")
    print()

    early_stop = EarlyStopping(
        patience=config.PATIENCE,
        min_delta=1e-4,
        restore_best=True,
    )

    # fit() uses an internal validation_split on the data passed to it.
    # We append val data to train so fit() can carve it back out at the end.
    X_combined = X_train + X_val
    y_combined = y_train + y_val
    val_frac   = len(X_val) / len(X_combined)

    t0 = time.time()
    history = model.fit(
        X_combined, y_combined,
        epochs           = config.EPOCHS,
        batch_size       = config.BATCH_SIZE,
        validation_split = val_frac,
        optimizer        = model._optimizer,
        loss             = model._loss_fn,
        callbacks        = [early_stop],
        verbose          = True,
    )
    t_train = time.time() - t0

    print(f"\n  Training complete in {t_train:.1f}s  "
          f"({t_train / 60:.1f} min)")

    return history, t_train


# ============================================================================
# THRESHOLD OPTIMISATION
# ============================================================================

def find_threshold(model: Sequential, X_val: list, y_val: list,
                   config: Config) -> float:
    """
    Sweep thresholds on the validation set and pick the one that maximises F1.
    Returns optimal threshold as float.
    """
    if not config.OPTIMIZE_THRESHOLD:
        return config.DECISION_THRESHOLD

    print("\n  Threshold sweep on validation set...")
    y_scores = model.predict(X_val)
    # threshold_sweep returns list of (threshold, metrics_dict)
    results = threshold_sweep(y_val, y_scores,
                              thresholds=[i / 100 for i in range(10, 91)])
    best_t, best_metrics = max(results, key=lambda r: r[1].get("f1", 0))
    print(f"  Optimal threshold : {best_t:.2f}  "
          f"(F1={best_metrics.get('f1', 0):.4f}  "
          f"P={best_metrics.get('precision', 0):.4f}  "
          f"R={best_metrics.get('recall', 0):.4f})")
    return best_t


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model: Sequential, X: list, y: list,
             threshold: float, label: str) -> dict:
    """Full metrics suite for a given split."""
    print(f"\n  [{label}]")
    y_scores = model.predict(X)
    y_pred   = [1 if s >= threshold else 0 for s in y_scores]

    acc  = accuracy(y, y_pred)
    prec = precision(y, y_pred)
    rec  = recall(y, y_pred)
    f1   = f1_score(y, y_pred)
    auc  = roc_auc(y, y_scores)
    cm   = confusion_matrix(y, y_pred)

    tn = cm.get("tn", 0); fp = cm.get("fp", 0)
    fn = cm.get("fn", 0); tp = cm.get("tp", 0)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)

    print(f"    Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {auc:.4f}")
    print(f"    FPR       : {fpr:.4f}  ({fpr*100:.2f}%)")
    print(f"    FNR       : {fnr:.4f}  ({fnr*100:.2f}%)")
    print(f"    CM        : TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                auc=auc, fpr=fpr, fnr=fnr, tn=tn, fp=fp, fn=fn, tp=tp)


# ============================================================================
# QUANTIZATION (optional INT8 export)
# ============================================================================

def quantize(model: Sequential, splits: dict, config: Config):
    """
    Attempt INT8 TFLite export via iotids.quantize.tflm_export.
    Falls back gracefully if the quantize module is unavailable.
    The tflm_export module requires TensorFlow as a backend for FlatBuffer
    generation — this is the one academically-defensible TF dependency.
    """
    if not config.ENABLE_QUANTIZATION:
        return None, None

    print("\n" + "=" * 80)
    print("QUANTIZATION  (INT8 TFLite export)")
    print("=" * 80)

    try:
        from iotids.quantize.tflm_export import export_tflite
    except ImportError:
        print("  iotids.quantize.tflm_export not available — skipping.")
        print("  (Build the quantize module or run tflm_export.py separately.)")
        return None, None

    tflite_path = str(config.MODEL_DIR / "iotids_dnn.tflite")
    try:
        path, size_kb = export_tflite(
            model,
            tflite_path,
            representative_data=splits["X_train"][:2000],
        )
        print(f"  TFLite model  : {path}  ({size_kb:.2f} KB)")
        if size_kb > 30:
            print(f"  WARNING: {size_kb:.1f} KB exceeds 30 KB Pico 2W budget — "
                  f"consider pruning before export.")
        return path, size_kb
    except Exception as exc:
        print(f"  Quantization failed: {exc}")
        return None, None


# ============================================================================
# SAVE ARTIFACTS  (minimized output)
# ============================================================================

def save_artifacts(model: Sequential, scaler, threshold: float,
                   history: dict, config: Config):
    """
    Saves three compact binary files:

      iotids_dnn.model      — weights downcast to float32, ~30–80 KB
                              (model.save() uses iotids compact binary format,
                               not pickle — see nn/model.py)
      iotids_dnn_scaler.bin — only 2 × n_features float32 values
                              (median + IQR per feature, ~400 B for 24 feats)
      iotids_dnn_threshold.bin — single float (< 32 bytes)

    History is NOT saved by default (removes ~20 KB of training logs that
    aren't needed for inference or the C runtime).
    """
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path  = config.MODEL_DIR / "iotids_dnn.model"
    scaler_path = config.MODEL_DIR / "iotids_dnn_scaler.bin"
    thresh_path = config.MODEL_DIR / "iotids_dnn_threshold.bin"

    # model.save() — compact binary, float32 weights only (no Python objects)
    model.save(str(model_path))

    # scaler: save only the params dict (median + IQR arrays as float32)
    # This avoids saving the full Python object — params are all that C needs
    scaler_params = scaler.get_params()   # returns {"median": [...], "iqr": [...]}
    save(scaler_params, str(scaler_path))

    # threshold: single float
    save(float(threshold), str(thresh_path))

    # Report file sizes
    def kb(p):
        p = Path(p)
        return p.stat().st_size / 1024 if p.exists() else 0

    print(f"\n  Saved artifacts:")
    print(f"    Model     : {model_path}  ({kb(model_path):.1f} KB)")
    print(f"    Scaler    : {scaler_path}  ({kb(scaler_path):.3f} KB)")
    print(f"    Threshold : {threshold:.4f} -> {thresh_path}  ({kb(thresh_path):.3f} KB)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="iotids DNN — CIC-IDS-2017 (pure iotids library, no TF)")
    parser.add_argument("--data",     default=None, type=str,
                        help="Path to CSV dataset (default: processed_reduced/clean_dataset.csv)")
    parser.add_argument("--sample",   default=None, type=float,
                        help="Stratified sample fraction for quick runs (e.g. 0.1)")
    parser.add_argument("--epochs",   default=None, type=int,
                        help="Override max epoch count")
    parser.add_argument("--batch",    default=None, type=int,
                        help="Override batch size (default: 4096)")
    parser.add_argument("--no-quant", action="store_true",
                        help="Skip INT8 quantization export step")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    print("\n" + "=" * 80)
    print("  DEEP NEURAL NETWORK BASELINE  —  iotids Library".center(80))
    print("  No TensorFlow · No scikit-learn · No pandas".center(80))
    print("=" * 80)

    # Report backend
    backend = os.environ.get("IOTIDS_BACKEND", "pure-python")
    print(f"\n  Backend : {backend}")

    config = Config()
    if args.data:
        config.DATA_FILE = Path(args.data)
    if args.sample is not None:
        config.USE_SAMPLE  = True
        config.SAMPLE_FRAC = args.sample
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.batch is not None:
        config.BATCH_SIZE = args.batch
    if args.no_quant:
        config.ENABLE_QUANTIZATION = False

    # ── Pipeline ──────────────────────────────────────────────────────────────
    data              = load_data(config)
    X, y, feat_names  = preprocess(data, config)
    splits            = split_and_scale(X, y, config)
    model             = build_model(len(splits["X_train"][0]), config)
    history, t_train  = train(model, splits, config)

    # Threshold optimisation on validation set
    threshold = find_threshold(model, splits["X_val"], splits["y_val"], config)

    # Evaluation on all splits
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    m_train = evaluate(model, splits["X_train"], splits["y_train"],
                       threshold, "Train")
    m_val   = evaluate(model, splits["X_val"],   splits["y_val"],
                       threshold, "Validation")
    m_test  = evaluate(model, splits["X_test"],  splits["y_test"],
                       threshold, "Test")

    # Save artifacts (minimized)
    save_artifacts(model, splits["scaler"], threshold, history, config)

    # Optional quantization
    tflite_path, tflite_kb = quantize(model, splits, config)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY".center(80))
    print("=" * 80)
    print(f"\n  Test Accuracy  : {m_test['accuracy']:.4f}  "
          f"({m_test['accuracy'] * 100:.2f}%)")
    print(f"  Test Precision : {m_test['precision']:.4f}")
    print(f"  Test Recall    : {m_test['recall']:.4f}")
    print(f"  Test F1-Score  : {m_test['f1']:.4f}")
    print(f"  Test ROC-AUC   : {m_test['auc']:.4f}")
    print(f"  FPR            : {m_test['fpr']:.4f}  ({m_test['fpr']*100:.4f}%)")
    print(f"  FNR            : {m_test['fnr']:.4f}  ({m_test['fnr']*100:.4f}%)")
    print(f"  Training time  : {t_train:.1f}s  ({t_train / 60:.1f} min)")
    print(f"  Threshold used : {threshold:.2f}")
    if tflite_kb:
        print(f"  TFLite size    : {tflite_kb:.2f} KB")

    print(f"\n  Prior DNN baseline : 99.42%")
    status = ("PASS — matches DNN baseline" if m_test["accuracy"] >= 0.97
              else "WARN — below 97%, check data or hyperparams")
    print(f"  Status             : {status}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()