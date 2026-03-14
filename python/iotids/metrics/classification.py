import math


def _check(y_true, y_pred):
    assert len(y_true) == len(y_pred), "Length mismatch"


def confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1
    return cm, labels


def accuracy(y_true, y_pred):
    _check(y_true, y_pred)
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def precision(y_true, y_pred, pos_label=1):
    tp = sum(1 for t, p in zip(y_true, y_pred) if p == pos_label and t == pos_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if p == pos_label and t != pos_label)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred, pos_label=1):
    tp = sum(1 for t, p in zip(y_true, y_pred) if p == pos_label and t == pos_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if p != pos_label and t == pos_label)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred, pos_label=1):
    p = precision(y_true, y_pred, pos_label)
    r = recall(y_true, y_pred, pos_label)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def roc_auc(y_true, y_scores):
    """
    Binary ROC-AUC via trapezoidal rule.
    y_true: list of 0/1, y_scores: list of floats.
    """
    pairs = sorted(zip(y_scores, y_true), key=lambda x: -x[0])
    total_pos = sum(y_true)
    total_neg = len(y_true) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5

    tp, fp = 0, 0
    prev_score = None
    prev_tp, prev_fp = 0, 0
    auc = 0.0

    for score, label in pairs:
        if score != prev_score and prev_score is not None:
            tpr_prev = prev_tp / total_pos
            fpr_prev = prev_fp / total_neg
            tpr_cur  = tp / total_pos
            fpr_cur  = fp / total_neg
            auc += 0.5 * (tpr_prev + tpr_cur) * abs(fpr_cur - fpr_prev)
            prev_tp, prev_fp = tp, fp
        prev_score = score
        if label == 1:
            tp += 1
        else:
            fp += 1

    # Final trapezoid
    tpr_prev = prev_tp / total_pos
    fpr_prev = prev_fp / total_neg
    auc += 0.5 * (tpr_prev + tp / total_pos) * abs(fp / total_neg - fpr_prev)
    return auc


def threshold_sweep(y_true, y_scores, thresholds=None):
    """
    Sweep thresholds, return list of (threshold, f1, precision, recall).
    Identifies optimal cutoff — critical after each compression stage.
    """
    if thresholds is None:
        thresholds = [i / 100 for i in range(5, 96, 5)]
    results = []
    for t in thresholds:
        y_pred = [1 if s >= t else 0 for s in y_scores]
        results.append((
            t,
            f1_score(y_true, y_pred),
            precision(y_true, y_pred),
            recall(y_true, y_pred),
        ))
    return results


def classification_report(y_true, y_pred, y_scores=None):
    lines = []
    acc = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    lines.append(f"  Accuracy : {acc:.4f}")
    lines.append(f"  Precision: {prec:.4f}")
    lines.append(f"  Recall   : {rec:.4f}")
    lines.append(f"  F1 Score : {f1:.4f}")
    if y_scores is not None:
        auc = roc_auc(y_true, y_scores)
        lines.append(f"  ROC-AUC  : {auc:.4f}")
    return "\n".join(lines)
