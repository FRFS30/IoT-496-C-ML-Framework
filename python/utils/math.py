import math


def percentile(x, q):
    """Return q-th percentile of iterable x (0-100)."""
    s = sorted(x)
    n = len(s)
    if n == 0:
        return 0.0
    idx = (q / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return s[lo] + frac * (s[hi] - s[lo])


def isnan(v):
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return True


def isinf(v):
    try:
        return math.isinf(v)
    except (TypeError, ValueError):
        return False


def nan_to_num(v, replacement=0.0):
    if isnan(v) or isinf(v):
        return replacement
    return v


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def log_safe(v, eps=1e-9):
    return math.log(max(v, eps))


def sigmoid(v):
    if v >= 0:
        return 1.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return e / (1.0 + e)


def col_medians(rows):
    """Return list of column medians from list-of-lists."""
    if not rows:
        return []
    ncols = len(rows[0])
    result = []
    for j in range(ncols):
        col = [rows[i][j] for i in range(len(rows)) if not isnan(rows[i][j])]
        result.append(percentile(col, 50) if col else 0.0)
    return result


def col_iqr(rows):
    """Return list of IQRs per column."""
    if not rows:
        return []
    ncols = len(rows[0])
    result = []
    for j in range(ncols):
        col = [rows[i][j] for i in range(len(rows)) if not isnan(rows[i][j])]
        if col:
            result.append(max(percentile(col, 75) - percentile(col, 25), 1e-8))
        else:
            result.append(1.0)
    return result
