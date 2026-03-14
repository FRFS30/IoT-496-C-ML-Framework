import array
import math
from .tensor import Tensor, _prod


# ------------------------------------------------------------------ #
# Elementwise binary ops with scalar or same-shape broadcasting
# ------------------------------------------------------------------ #
def _ew(a, b, fn):
    ad = a.data if isinstance(a, Tensor) else None
    bd = b.data if isinstance(b, Tensor) else None
    if ad is None:          # a is scalar
        out = array.array("f", (fn(a, v) for v in bd))
        return Tensor(out, b.shape)
    if bd is None:          # b is scalar
        out = array.array("f", (fn(v, b) for v in ad))
        return Tensor(out, a.shape)
    assert a.shape == b.shape, f"Shape mismatch {a.shape} vs {b.shape}"
    out = array.array("f", (fn(x, y) for x, y in zip(ad, bd)))
    return Tensor(out, a.shape)


def add(a, b): return _ew(a, b, lambda x, y: x + y)
def sub(a, b): return _ew(a, b, lambda x, y: x - y)
def mul(a, b): return _ew(a, b, lambda x, y: x * y)
def div(a, b): return _ew(a, b, lambda x, y: x / y if y != 0.0 else 0.0)


# ------------------------------------------------------------------ #
# Matrix multiply  (critical hot path)
# ------------------------------------------------------------------ #
def dot(a, b):
    """Optimised 2-D matmul. a: (m,k), b: (k,n) -> (m,n)."""
    assert len(a.shape) == 2 and len(b.shape) == 2
    m, k = a.shape
    k2, n = b.shape
    assert k == k2, f"dot: inner dims {k} != {k2}"

    ad, bd = a.data, b.data
    out = array.array("f", [0.0] * (m * n))

    # Transpose b for cache-friendly access
    bt = array.array("f", [0.0] * (k * n))
    for i in range(k):
        for j in range(n):
            bt[j * k + i] = bd[i * n + j]

    for i in range(m):
        row_off = i * k
        for j in range(n):
            col_off = j * k
            s = 0.0
            for p in range(k):
                s += ad[row_off + p] * bt[col_off + p]
            out[i * n + j] = s
    return Tensor(out, (m, n))


# ------------------------------------------------------------------ #
# Unary ops
# ------------------------------------------------------------------ #
def relu(x):
    return Tensor(array.array("f", (v if v > 0.0 else 0.0 for v in x.data)), x.shape)


def leaky_relu(x, alpha=0.01):
    return Tensor(array.array("f", (v if v > 0.0 else alpha * v for v in x.data)), x.shape)


def sigmoid(x):
    def _sig(v):
        if v >= 0:
            return 1.0 / (1.0 + math.exp(-v))
        e = math.exp(v)
        return e / (1.0 + e)
    return Tensor(array.array("f", (_sig(v) for v in x.data)), x.shape)


def softmax(x):
    d = x.data
    mx = max(d)
    exps = [math.exp(v - mx) for v in d]
    s = sum(exps)
    return Tensor(array.array("f", (e / s for e in exps)), x.shape)


def clip(x, lo, hi):
    return Tensor(array.array("f", (max(lo, min(hi, v)) for v in x.data)), x.shape)


def abs_(x):
    return Tensor(array.array("f", (abs(v) for v in x.data)), x.shape)


def log_(x):
    eps = 1e-9
    return Tensor(array.array("f", (math.log(v + eps) for v in x.data)), x.shape)


def exp_(x):
    return Tensor(array.array("f", (math.exp(v) for v in x.data)), x.shape)


# ------------------------------------------------------------------ #
# Reductions with optional axis
# ------------------------------------------------------------------ #
def _reduce(x, fn_init, fn_accum, axis=None):
    if axis is None:
        acc = fn_init(x.data[0])
        for v in x.data[1:]:
            acc = fn_accum(acc, v)
        return acc
    assert len(x.shape) == 2
    r, c = x.shape
    if axis == 0:           # reduce rows -> shape (c,)
        out = [fn_init(x.data[j]) for j in range(c)]
        for i in range(1, r):
            for j in range(c):
                out[j] = fn_accum(out[j], x.data[i * c + j])
        return Tensor(out, (c,))
    if axis == 1:           # reduce cols -> shape (r,)
        out = []
        for i in range(r):
            row = x.data[i * c: i * c + c]
            acc = fn_init(row[0])
            for v in row[1:]:
                acc = fn_accum(acc, v)
            out.append(acc)
        return Tensor(out, (r,))
    raise ValueError(f"Unsupported axis {axis}")


def sum_(x, axis=None):
    return _reduce(x, lambda v: v, lambda a, b: a + b, axis)


def max_(x, axis=None):
    return _reduce(x, lambda v: v, max, axis)


def argmax(x, axis=None):
    if axis is None:
        return x.data.index(max(x.data))
    assert len(x.shape) == 2
    r, c = x.shape
    if axis == 1:
        out = []
        for i in range(r):
            row = x.data[i * c: i * c + c]
            mx = max(row)
            out.append(row.index(mx))
        return Tensor(array.array("f", out), (r,))
    raise ValueError("argmax axis=0 not implemented")


def mean(x, axis=None):
    s = sum_(x, axis)
    if axis is None:
        return s / len(x.data)
    n = x.shape[1] if axis == 1 else x.shape[0]
    return div(s, float(n))


def std(x, axis=None):
    m = mean(x, axis)
    if axis is None:
        diff = [(v - m) ** 2 for v in x.data]
        return math.sqrt(sum(diff) / len(diff))
    # axis-wise variance
    r, c = x.shape
    if axis == 0:
        md = m.data
        var = [0.0] * c
        for i in range(r):
            for j in range(c):
                var[j] += (x.data[i * c + j] - md[j]) ** 2
        return Tensor(array.array("f", (math.sqrt(v / r) for v in var)), (c,))
    raise ValueError("std axis=1 not implemented")


def var(x, axis=None):
    m = mean(x, axis)
    if axis is None:
        return sum((v - m) ** 2 for v in x.data) / len(x.data)
    raise ValueError("var with axis not implemented")
