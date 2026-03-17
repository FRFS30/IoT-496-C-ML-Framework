"""
CIC-IDS-2017 Deep Neural Network Baseline — iotids Library
===========================================================
Trains, evaluates, prunes, quantizes, and saves a DNN classifier using only
the iotids Python library + Python stdlib. No TensorFlow, scikit-learn, or
pandas dependency at any stage.

Pipeline
--------
  1. Load + preprocess CIC-IDS-2017 (full or sampled)
  2. Train DNN with weighted BCE + Adam + EarlyStopping
  3. Evaluate on train/val/test splits
  4. Magnitude pruning (zeroes bottom-N% weights, fine-tunes 3 epochs)
  5. INT8 post-training quantization (per-layer symmetric)
  6. Save compact .bin files (pure struct — no pickle, no external deps)

Output files  (models/iotids_dnn/)
------------------------------------
  iotids_dnn.bin        — pruned + INT8-quantized weights, pure binary
  iotids_dnn_f32.bin    — float32 pruned weights (C runtime fallback)
  iotids_dnn_scaler.bin — RobustScaler params (median + IQR as float32)
  iotids_dnn_threshold.bin — optimal decision threshold (float32)

Run
---
  nohup python3 -u dnn.py --data processed_reduced/clean_dataset.csv > dnn.log 2>&1 &
  python3 dnn.py --data processed_reduced/clean_dataset.csv --sample 0.1
  python3 dnn.py --data processed_reduced/clean_dataset.csv --no-compress
"""

import argparse
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PYTHON_DIR = _HERE / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from iotids.data.csv_reader    import read_csv
from iotids.data.preprocessing import (
    RobustScaler, clip_outliers, replace_inf, drop_nan_rows,
)
from iotids.data.dataset       import Dataset
from iotids.nn.layers          import Dense, BatchNormalization, Dropout
from iotids.nn.losses          import BinaryCrossentropy
from iotids.nn.optimizers      import Adam
from iotids.nn.model           import Sequential, EarlyStopping, LRScheduler
from iotids.metrics.classification import (
    accuracy, precision, recall, f1_score, roc_auc,
    confusion_matrix, threshold_sweep,
)
from iotids.utils.io     import save, load
from iotids.utils.random import set_seed

RANDOM_SEED = 42


# ============================================================================
# WEIGHTED BCE  (in-place gradient scaling — no library changes needed)
# ============================================================================

class WeightedBCE:
    """
    Vectorized binary cross-entropy with per-sample class weighting.

    Fully numpy — no Python loops over samples. This gives stable,
    numerically clean gradients and is the primary fix for the ~96.5%
    accuracy ceiling caused by the old pure-Python implementation.

    from_logits=True: expects raw logits; applies numerically stable
    log-sum-exp sigmoid internally to avoid overflow on large logits.
    """

    def __init__(self, pos_weight: float = 3.0, from_logits: bool = True):
        self.pos_weight  = pos_weight
        self.from_logits = from_logits

    @staticmethod
    def _sigmoid(z):
        # Numerically stable sigmoid: avoids exp overflow for large |z|
        return np.where(z >= 0,
                        1.0 / (1.0 + np.exp(-z)),
                        np.exp(z) / (1.0 + np.exp(z)))

    def __call__(self, y_true, y_pred):
        eps = 1e-9
        yt  = np.asarray(y_true,  dtype=np.float64)
        yp  = np.asarray(y_pred,  dtype=np.float64)
        p   = self._sigmoid(yp) if self.from_logits else np.clip(yp, eps, 1 - eps)
        p   = np.clip(p, eps, 1 - eps)
        w   = np.where(yt == 1, self.pos_weight, 1.0)
        loss = -w * (yt * np.log(p) + (1.0 - yt) * np.log(1.0 - p))
        return float(loss.mean())

    def gradient(self, y_true, y_pred):
        yt = np.asarray(y_true, dtype=np.float64)
        yp = np.asarray(y_pred, dtype=np.float64)
        w  = np.where(yt == 1, self.pos_weight, 1.0)
        if self.from_logits:
            p     = self._sigmoid(yp)
            grads = w * (p - yt) / len(yt)
        else:
            eps   = 1e-9
            p     = np.clip(yp, eps, 1 - eps)
            grads = w * (-yt / p + (1.0 - yt) / (1.0 - p)) / len(yt)
        return grads.tolist()


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

    # Training
    EPOCHS        = 80
    BATCH_SIZE    = 4096
    LEARNING_RATE = 1e-3          # higher initial LR — scheduler will decay it
    PATIENCE      = 15            # was 10 — gives more room to escape plateaus

    # LR step decay: lr = lr0 * 0.5^floor(epoch / 15)
    LR_DROP       = 0.5
    LR_DROP_EVERY = 15

    # Architecture — wider first layer for more capacity
    HIDDEN_UNITS  = [512, 256, 128]   # was [256, 128, 64]
    DROPOUT_RATES = [0.4, 0.4, 0.3]

    # Decision threshold
    DECISION_THRESHOLD = 0.4
    OPTIMIZE_THRESHOLD = True

    # Oversampling
    USE_OVERSAMPLING  = True
    OVERSAMPLE_RATIO  = 0.5

    # Pruning
    ENABLE_PRUNING        = True
    PRUNE_SPARSITY        = 0.40
    PRUNE_FINETUNE_EPOCHS = 3

    # Quantization
    ENABLE_QUANTIZATION = True

    # Columns to exclude
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
        raise FileNotFoundError(f"\n  Dataset not found: {config.DATA_FILE}")

    print(f"\n  Loading: {config.DATA_FILE}")
    data   = read_csv(str(config.DATA_FILE))
    n_rows = len(next(iter(data.values())))
    print(f"  Loaded {n_rows:,} rows, {len(data)} columns")

    labels = data.get("Label", data.get(" Label", []))
    counts: dict = {}
    for v in labels:
        counts[v] = counts.get(v, 0) + 1
    print("  Class distribution:")
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {str(cls):<40} {cnt:>10,}  ({cnt / n_rows * 100:.2f}%)")

    if config.USE_SAMPLE:
        frac = config.SAMPLE_FRAC
        print(f"  Stratified {int(frac * 100)}% sample...")
        by_class: dict = {}
        for i, v in enumerate(labels):
            by_class.setdefault(v, []).append(i)
        import random as _rnd
        _rnd.seed(RANDOM_SEED)
        keep: set = set()
        for cls_indices in by_class.values():
            k = max(1, int(len(cls_indices) * frac))
            keep.update(_rnd.sample(cls_indices, k))
        data   = {col: [v for i, v in enumerate(vals) if i in keep]
                  for col, vals in data.items()}
        n_rows = len(next(iter(data.values())))
        print(f"  Sample size: {n_rows:,}")

    return data


# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess(data: dict, config: Config):
    print("\n" + "=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)

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

    print("  Converting to float arrays...")
    X_raw: list = []
    for i in range(len(data[feat_names[0]])):
        row = []
        for col in feat_names:
            try:
                row.append(float(data[col][i]))
            except (ValueError, TypeError):
                row.append(float("nan"))
        X_raw.append(row)

    raw_labels = data[label_key]

    print("  Replacing Inf values...")
    X_raw = replace_inf(X_raw)
    print("  Dropping NaN rows...")
    X_raw, raw_labels = drop_nan_rows(X_raw, raw_labels)
    print("  Clipping outliers [1st, 99th pct]...")
    X_raw = clip_outliers(X_raw, low_pct=1, high_pct=99)

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
# SPLIT, SCALE, OVERSAMPLE
# ============================================================================

def split_and_scale(X: list, y: list, config: Config) -> dict:
    print("\n" + "=" * 80)
    print("DATA SPLITTING AND SCALING")
    print("=" * 80)

    ds = Dataset(X, y)
    trainval_ds, test_ds = ds.train_test_split(
        test_size=config.TEST_SIZE, stratify=True, seed=RANDOM_SEED)
    val_frac = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    train_ds, val_ds = Dataset(trainval_ds.X, trainval_ds.y).train_test_split(
        test_size=val_frac, stratify=True, seed=RANDOM_SEED)

    print(f"  Train : {len(train_ds.X):,}  ({(1-config.TEST_SIZE-config.VAL_SIZE)*100:.1f}%)")
    print(f"  Val   : {len(val_ds.X):,}  ({config.VAL_SIZE*100:.1f}%)")
    print(f"  Test  : {len(test_ds.X):,}  ({config.TEST_SIZE*100:.1f}%)")

    print("  Fitting RobustScaler on training data...")
    scaler         = RobustScaler()
    X_train_scaled = scaler.fit_transform(train_ds.X)
    X_val_scaled   = scaler.transform(val_ds.X)
    X_test_scaled  = scaler.transform(test_ds.X)
    print("  Scaling complete.")

    y_train = list(train_ds.y)

    if config.USE_OVERSAMPLING:
        benign_idx      = [i for i, v in enumerate(y_train) if v == 0]
        attack_idx      = [i for i, v in enumerate(y_train) if v == 1]
        target_n_attack = int(len(benign_idx) * config.OVERSAMPLE_RATIO)
        if target_n_attack > len(attack_idx):
            import random as _rnd
            _rnd.seed(RANDOM_SEED)
            extra          = _rnd.choices(attack_idx,
                                          k=target_n_attack - len(attack_idx))
            X_train_scaled = X_train_scaled + [X_train_scaled[i] for i in extra]
            y_train        = y_train + [1] * len(extra)
            print(f"  Oversampled: {len(attack_idx):,} → {target_n_attack:,} attacks"
                  f"  (total train: {len(y_train):,})")

    return {
        "X_train": X_train_scaled, "y_train": y_train,
        "X_val":   X_val_scaled,   "y_val":   list(val_ds.y),
        "X_test":  X_test_scaled,  "y_test":  list(test_ds.y),
        "scaler":  scaler,
    }


# ============================================================================
# MODEL BUILD
# ============================================================================

def build_model(n_features: int, config: Config,
                pos_weight: float = 2.0) -> Sequential:
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE  (iotids.nn)")
    print("=" * 80)

    layers  = []
    in_size = n_features
    for units, drop in zip(config.HIDDEN_UNITS, config.DROPOUT_RATES):
        layers.append(Dense(in_size, units, activation="relu"))
        layers.append(BatchNormalization(units))
        layers.append(Dropout(drop))
        in_size = units
    layers.append(Dense(in_size, 1, activation=None))

    model              = Sequential(layers)
    model._loss_fn     = WeightedBCE(pos_weight=pos_weight, from_logits=True)
    model._optimizer   = Adam(lr=config.LEARNING_RATE)

    print(f"\n  Loss     : WeightedBCE(pos_weight={pos_weight:.2f}, from_logits=True)")
    print(f"  Optimizer: Adam(lr={config.LEARNING_RATE})")
    print(f"  Arch     : {n_features} -> " +
          " -> ".join(str(u) for u in config.HIDDEN_UNITS) + " -> 1")
    print(f"  Dropout  : {config.DROPOUT_RATES}")
    print(f"  Batch    : {config.BATCH_SIZE}")
    print(f"  LR decay : drop={config.LR_DROP} every={config.LR_DROP_EVERY} epochs")
    return model


# ============================================================================
# TRAINING
# ============================================================================

def train(model: Sequential, splits: dict, config: Config,
          extra_epochs: int = None, tag: str = "TRAINING"):
    print("\n" + "=" * 80)
    print(tag)
    print("=" * 80)

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val   = splits["X_val"]
    y_val   = splits["y_val"]
    epochs  = extra_epochs if extra_epochs is not None else config.EPOCHS

    print(f"\n  Training samples   : {len(X_train):,}")
    print(f"  Validation samples : {len(X_val):,}")
    print(f"  Batch size         : {config.BATCH_SIZE}")
    print(f"  Steps/epoch        : {max(1, len(X_train) // config.BATCH_SIZE)}")
    print(f"  Max epochs         : {epochs}")

    early_stop = EarlyStopping(patience=config.PATIENCE, min_delta=1e-4,
                               restore_best=True)
    lr_sched   = LRScheduler(model._optimizer,
                              drop=config.LR_DROP,
                              every=config.LR_DROP_EVERY)

    X_combined = X_train + X_val
    y_combined = y_train + y_val
    val_frac   = len(X_val) / len(X_combined)

    t0 = time.time()
    history = model.fit(
        X_combined, y_combined,
        epochs           = epochs,
        batch_size       = config.BATCH_SIZE,
        validation_split = val_frac,
        optimizer        = model._optimizer,
        loss             = model._loss_fn,
        callbacks        = [early_stop, lr_sched],
        verbose          = True,
    )
    t_train = time.time() - t0
    print(f"\n  Done in {t_train:.1f}s  ({t_train / 60:.1f} min)")
    return history, t_train


# ============================================================================
# PRUNING
# ============================================================================

def prune_model(model: Sequential, splits: dict, config: Config) -> float:
    print("\n" + "=" * 80)
    print("PRUNING  (magnitude-based)")
    print("=" * 80)

    all_magnitudes = []
    for layer in model.layers:
        if isinstance(layer, Dense):
            flat_W, _ = layer.get_weights()
            for w in flat_W:
                all_magnitudes.append(abs(w))

    all_magnitudes.sort()
    threshold_idx = int(len(all_magnitudes) * config.PRUNE_SPARSITY)
    mag_threshold = all_magnitudes[threshold_idx]

    print(f"\n  Total Dense weights : {len(all_magnitudes):,}")
    print(f"  Sparsity target     : {config.PRUNE_SPARSITY * 100:.0f}%")
    print(f"  Magnitude threshold : {mag_threshold:.6f}")

    zeroed = 0
    for layer in model.layers:
        if isinstance(layer, Dense):
            flat_W, b = layer.get_weights()
            flat_W    = [0.0 if abs(w) < mag_threshold else w for w in flat_W]
            zeroed   += sum(1 for w in flat_W if w == 0.0)
            layer.set_weights([flat_W, b])

    actual_sparsity = zeroed / max(len(all_magnitudes), 1)
    print(f"  Weights zeroed      : {zeroed:,}  (actual {actual_sparsity*100:.1f}%)")

    if config.PRUNE_FINETUNE_EPOCHS > 0:
        print(f"\n  Fine-tuning {config.PRUNE_FINETUNE_EPOCHS} epoch(s) after pruning...")
        orig_lr              = model._optimizer.lr
        model._optimizer.lr  = orig_lr * 0.2
        train(model, splits, config,
              extra_epochs=config.PRUNE_FINETUNE_EPOCHS,
              tag="PRUNING FINE-TUNE")
        model._optimizer.lr  = orig_lr

    return actual_sparsity


# ============================================================================
# INT8 QUANTIZATION
# ============================================================================

def _compute_scale_zp(values: list):
    max_abs = max(abs(v) for v in values) if values else 1.0
    if max_abs == 0.0:
        max_abs = 1.0
    return max_abs / 127.0, 0


def _quantize_array(values: list, scale: float, zp: int) -> list:
    return [max(-128, min(127, int(round(v / scale)) + zp)) for v in values]


def quantize_model(model: Sequential) -> dict:
    print("\n" + "=" * 80)
    print("QUANTIZATION  (INT8 per-layer symmetric)")
    print("=" * 80)

    quant_data       = {}
    total_bytes_f32  = 0
    total_bytes_int8 = 0

    for idx, layer in enumerate(model.layers):
        if isinstance(layer, Dense):
            flat_W, b     = layer.get_weights()
            scale_W, zp_W = _compute_scale_zp(flat_W)
            W_q           = _quantize_array(flat_W, scale_W, zp_W)
            scale_b, zp_b = _compute_scale_zp(b) if b else (1.0, 0)
            b_q           = _quantize_array(b, scale_b, zp_b) if b else []

            quant_data[idx] = {
                "type": "Dense",
                "in_features": layer.in_features,
                "units":       layer.units,
                "scale_W": scale_W, "zp_W": zp_W, "W_q": W_q,
                "scale_b": scale_b, "zp_b": zp_b, "b_q": b_q,
            }

            f32_bytes         = (len(flat_W) + len(b)) * 4
            int8_bytes        = len(W_q) + len(b_q) + 8
            total_bytes_f32  += f32_bytes
            total_bytes_int8 += int8_bytes
            print(f"  Layer {idx:2d} Dense({layer.in_features},{layer.units})"
                  f"  scale_W={scale_W:.6f}"
                  f"  {f32_bytes/1024:.1f}KB → {int8_bytes/1024:.1f}KB"
                  f"  ({f32_bytes/max(int8_bytes,1):.1f}x)")

        elif isinstance(layer, BatchNormalization):
            quant_data[idx] = {"type": "BatchNorm",
                               "weights": layer.get_weights(),
                               "features": layer.features}

        elif isinstance(layer, Dropout):
            quant_data[idx] = {"type": "Dropout", "rate": layer.rate}

    print(f"\n  Total: f32={total_bytes_f32/1024:.1f} KB"
          f"  →  int8={total_bytes_int8/1024:.1f} KB"
          f"  ({total_bytes_f32/max(total_bytes_int8,1):.1f}x compression)")
    return quant_data


# ============================================================================
# BINARY SAVE  (pure struct — no pickle, C-readable)
# ============================================================================
# Header:  [u16 magic=0xD1D5][i32 n_layers]
# Dense:   [u8=0][i32 in_f][i32 units][f32 scale_W][bxxx zp_W]
#          [i8 × in_f*units W_q][f32 scale_b][bxxx zp_b][i8 × units b_q]
# BN:      [u8=1][i32 features][f32×features gamma,beta,rmean,rvar]
# Dropout: [u8=2][f32 rate]

MAGIC = 0xD1D5

def _pack_f32(v):       return struct.pack("<f", v)
def _pack_f32_arr(arr): return struct.pack(f"<{len(arr)}f", *arr)
def _pack_i8_arr(arr):  return struct.pack(f"<{len(arr)}b", *arr)


def save_bin(quant_data: dict, path: str) -> float:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<HI", MAGIC, len(quant_data)))
        for idx in sorted(quant_data.keys()):
            d = quant_data[idx]
            if d["type"] == "Dense":
                f.write(struct.pack("<B", 0))
                f.write(struct.pack("<ii", d["in_features"], d["units"]))
                f.write(_pack_f32(d["scale_W"]))
                f.write(struct.pack("<bxxx", d["zp_W"]))
                f.write(_pack_i8_arr(d["W_q"]))
                f.write(_pack_f32(d["scale_b"]))
                f.write(struct.pack("<bxxx", d["zp_b"]))
                f.write(_pack_i8_arr(d["b_q"]))
            elif d["type"] == "BatchNorm":
                f.write(struct.pack("<B", 1))
                f.write(struct.pack("<i", d["features"]))
                for arr in d["weights"][:4]:
                    f.write(_pack_f32_arr(arr))
            elif d["type"] == "Dropout":
                f.write(struct.pack("<B", 2))
                f.write(_pack_f32(d["rate"]))
    size_kb = path.stat().st_size / 1024
    print(f"  INT8 .bin : {path}  ({size_kb:.2f} KB)")
    return size_kb


def save_f32_bin(model: Sequential, path: str) -> float:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<HI", MAGIC, len(model.layers)))
        for layer in model.layers:
            if isinstance(layer, Dense):
                flat_W, b = layer.get_weights()
                f.write(struct.pack("<B", 0))
                f.write(struct.pack("<ii", layer.in_features, layer.units))
                f.write(_pack_f32_arr(flat_W))
                f.write(_pack_f32_arr(b if b else []))
            elif isinstance(layer, BatchNormalization):
                f.write(struct.pack("<B", 1))
                f.write(struct.pack("<i", layer.features))
                for arr in layer.get_weights()[:4]:
                    f.write(_pack_f32_arr(arr))
            elif isinstance(layer, Dropout):
                f.write(struct.pack("<B", 2))
                f.write(_pack_f32(layer.rate))
    size_kb = path.stat().st_size / 1024
    print(f"  f32 .bin  : {path}  ({size_kb:.2f} KB)")
    return size_kb


def save_scaler_bin(scaler, path: str):
    path   = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = scaler.get_params()
    median = params["median"]
    iqr    = params["iqr"]
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(median)))
        f.write(_pack_f32_arr(median))
        f.write(_pack_f32_arr(iqr))
    print(f"  scaler    : {path}  ({path.stat().st_size / 1024:.3f} KB)")


def save_threshold_bin(threshold: float, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(_pack_f32(threshold))
    print(f"  threshold : {threshold:.4f} → {path}  ({path.stat().st_size} bytes)")


# ============================================================================
# THRESHOLD SWEEP
# ============================================================================

def find_threshold(model: Sequential, X_val: list, y_val: list,
                   config: Config) -> float:
    """
    threshold_sweep(y_true, y_scores, thresholds) returns:
        list of (threshold, f1, precision, recall) 4-tuples

    Index layout: r[0]=threshold, r[1]=f1, r[2]=precision, r[3]=recall
    """
    if not config.OPTIMIZE_THRESHOLD:
        return config.DECISION_THRESHOLD

    print("\n  Threshold sweep on validation set...")
    y_scores = model.predict(X_val)
    results  = threshold_sweep(y_val, y_scores,
                               thresholds=[i / 100 for i in range(10, 91)])

    # results is list of (threshold, f1, precision, recall) 4-tuples
    best = max(results, key=lambda r: r[1])
    best_t, best_f1, best_p, best_r = best[0], best[1], best[2], best[3]

    print(f"  Optimal threshold : {best_t:.2f}  "
          f"(F1={best_f1:.4f}  "
          f"P={best_p:.4f}  "
          f"R={best_r:.4f})")
    return float(best_t)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model: Sequential, X: list, y: list,
             threshold: float, label: str) -> dict:
    """
    confusion_matrix(y_true, y_pred) returns (cm_2d_list, labels).
    For binary classification: cm[0][0]=TN, cm[0][1]=FP, cm[1][0]=FN, cm[1][1]=TP
    """
    print(f"\n  [{label}]")
    y_scores = model.predict(X)
    y_pred   = [1 if s >= threshold else 0 for s in y_scores]

    acc  = accuracy(y, y_pred)
    prec = precision(y, y_pred)
    rec  = recall(y, y_pred)
    f1   = f1_score(y, y_pred)
    auc  = roc_auc(y, y_scores)

    # confusion_matrix returns (2d_list, labels) — unpack accordingly
    cm, _ = confusion_matrix(y, y_pred)
    tn = int(cm[0][0])
    fp = int(cm[0][1])
    fn = int(cm[1][0])
    tp = int(cm[1][1])

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
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="iotids DNN baseline")
    parser.add_argument("--data",        default=None,  type=str)
    parser.add_argument("--sample",      default=None,  type=float)
    parser.add_argument("--epochs",      default=None,  type=int)
    parser.add_argument("--batch",       default=None,  type=int)
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument("--sparsity",    default=None,  type=float)
    parser.add_argument("--pos-weight",  default=None,  type=float,
                        help="Class weight for attack samples (default: auto)")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    print("\n" + "=" * 80)
    print("  DEEP NEURAL NETWORK BASELINE  —  iotids Library".center(80))
    print("  No TensorFlow · No scikit-learn · No pandas".center(80))
    print("=" * 80)
    print(f"\n  Backend : {os.environ.get('IOTIDS_BACKEND', 'numpy')}")

    config = Config()
    if args.data:        config.DATA_FILE    = Path(args.data)
    if args.sample:      config.USE_SAMPLE   = True; config.SAMPLE_FRAC = args.sample
    if args.epochs:      config.EPOCHS       = args.epochs
    if args.batch:       config.BATCH_SIZE   = args.batch
    if args.no_compress:
        config.ENABLE_PRUNING      = False
        config.ENABLE_QUANTIZATION = False
    if args.sparsity:    config.PRUNE_SPARSITY = args.sparsity

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    data             = load_data(config)
    X, y, feat_names = preprocess(data, config)
    splits           = split_and_scale(X, y, config)

    # ── 2. Compute pos_weight ─────────────────────────────────────────────────
    # pos_weight=3.0: dataset is 80/20 benign/attack. After oversampling to
    # 0.5 ratio the minority class is still under-represented in loss terms.
    # 3.0 matches the pre-oversample ratio more closely and pushes recall up.
    pos_weight = args.pos_weight if args.pos_weight else 3.0
    print(f"\n  Class weight (pos_weight) : {pos_weight:.3f}x")

    # ── 3. Build + train ──────────────────────────────────────────────────────
    model            = build_model(len(splits["X_train"][0]), config, pos_weight)
    history, t_train = train(model, splits, config)

    # ── 4. Threshold + pre-compression eval ───────────────────────────────────
    threshold = find_threshold(model, splits["X_val"], splits["y_val"], config)

    print("\n" + "=" * 80)
    print("EVALUATION  (pre-compression)")
    print("=" * 80)
    evaluate(model, splits["X_train"], splits["y_train"], threshold, "Train")
    evaluate(model, splits["X_val"],   splits["y_val"],   threshold, "Validation")
    m_pre = evaluate(model, splits["X_test"], splits["y_test"], threshold, "Test")

    # ── 5. Pruning ────────────────────────────────────────────────────────────
    sparsity = 0.0
    if config.ENABLE_PRUNING:
        sparsity  = prune_model(model, splits, config)
        threshold = find_threshold(model, splits["X_val"], splits["y_val"], config)

        print("\n" + "=" * 80)
        print("EVALUATION  (post-pruning)")
        print("=" * 80)
        evaluate(model, splits["X_train"], splits["y_train"], threshold, "Train")
        evaluate(model, splits["X_val"],   splits["y_val"],   threshold, "Validation")
        m_post    = evaluate(model, splits["X_test"], splits["y_test"], threshold, "Test")
        acc_drop  = m_pre["accuracy"] - m_post["accuracy"]
        print(f"\n  Accuracy drop from pruning : {acc_drop*100:.3f}%")

    # ── 6. Save float32 .bin ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SAVING ARTIFACTS")
    print("=" * 80)
    save_f32_bin(model,          str(config.MODEL_DIR / "iotids_dnn_f32.bin"))
    save_scaler_bin(splits["scaler"], str(config.MODEL_DIR / "iotids_dnn_scaler.bin"))
    save_threshold_bin(threshold, str(config.MODEL_DIR / "iotids_dnn_threshold.bin"))

    # ── 7. INT8 quantization + save ──────────────────────────────────────────
    int8_kb = None
    if config.ENABLE_QUANTIZATION:
        quant_data = quantize_model(model)
        int8_kb    = save_bin(quant_data, str(config.MODEL_DIR / "iotids_dnn.bin"))

    # ── 8. Final evaluation ───────────────────────────────────────────────────
    m_test = evaluate(model, splits["X_test"], splits["y_test"],
                      threshold, "Final Test")

    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY".center(80))
    print("=" * 80)
    print(f"\n  Test Accuracy  : {m_test['accuracy']:.4f}  ({m_test['accuracy']*100:.2f}%)")
    print(f"  Test Precision : {m_test['precision']:.4f}")
    print(f"  Test Recall    : {m_test['recall']:.4f}")
    print(f"  Test F1-Score  : {m_test['f1']:.4f}")
    print(f"  Test ROC-AUC   : {m_test['auc']:.4f}")
    print(f"  FPR            : {m_test['fpr']:.4f}  ({m_test['fpr']*100:.4f}%)")
    print(f"  FNR            : {m_test['fnr']:.4f}  ({m_test['fnr']*100:.4f}%)")
    print(f"  Training time  : {t_train:.1f}s  ({t_train/60:.1f} min)")
    print(f"  Threshold used : {threshold:.2f}")
    print(f"  pos_weight     : {pos_weight:.3f}x")
    print(f"  Pruning        : {'OFF' if not config.ENABLE_PRUNING else f'{sparsity*100:.1f}% sparsity'}")
    if int8_kb:
        print(f"  INT8 .bin size : {int8_kb:.2f} KB"
              f"  ({'PASS' if int8_kb <= 30 else 'WARN: exceeds 30KB Pico budget'})")
    print(f"\n  Prior DNN baseline : 99.42%")
    print(f"  Status             : {'PASS' if m_test['accuracy'] >= 0.99 else 'WARN — below 99%'}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()