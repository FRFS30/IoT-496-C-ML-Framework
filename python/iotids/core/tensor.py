import array
import math
from .dtypes import INT8_MIN, INT8_MAX, cast_val, FLOAT32, INT8


class Tensor:
    """Flat float32 array with shape metadata. No external deps."""

    __slots__ = ("data", "shape")

    def __init__(self, data, shape=None):
        if isinstance(data, array.array):
            self.data = data
        else:
            self.data = array.array("f", (float(v) for v in data))
        n = len(self.data)
        if shape is None:
            self.shape = (n,)
        else:
            assert _prod(shape) == n, f"Shape {shape} != data length {n}"
            self.shape = tuple(shape)

    # ------------------------------------------------------------------ #
    # Factory methods
    # ------------------------------------------------------------------ #
    @staticmethod
    def zeros(shape):
        n = _prod(shape)
        return Tensor(array.array("f", [0.0] * n), shape)

    @staticmethod
    def ones(shape):
        n = _prod(shape)
        return Tensor(array.array("f", [1.0] * n), shape)

    @staticmethod
    def rand(shape, rng):
        n = _prod(shape)
        return Tensor([rng.random() for _ in range(n)], shape)

    @staticmethod
    def randn(shape, rng):
        n = _prod(shape)
        out = []
        for _ in range(n):
            out.append(rng.gauss(0.0, 1.0))
        return Tensor(out, shape)

    # ------------------------------------------------------------------ #
    # Shape ops
    # ------------------------------------------------------------------ #
    def reshape(self, new_shape):
        assert _prod(new_shape) == len(self.data)
        return Tensor(self.data, new_shape)

    def flatten(self):
        return Tensor(self.data, (len(self.data),))

    def transpose(self):
        """2-D only."""
        assert len(self.shape) == 2
        r, c = self.shape
        out = array.array("f", [0.0] * (r * c))
        for i in range(r):
            for j in range(c):
                out[j * r + i] = self.data[i * c + j]
        return Tensor(out, (c, r))

    # ------------------------------------------------------------------ #
    # Indexing / slicing
    # ------------------------------------------------------------------ #
    def __getitem__(self, idx):
        """Row slicing for 2-D tensors; element access for 1-D."""
        if len(self.shape) == 1:
            return self.data[idx]
        if len(self.shape) == 2:
            r, c = self.shape
            if isinstance(idx, int):
                if idx < 0:
                    idx += r
                start = idx * c
                return Tensor(self.data[start: start + c], (c,))
            if isinstance(idx, slice):
                rows = range(*idx.indices(r))
                out = array.array("f")
                for i in rows:
                    out.extend(self.data[i * c: i * c + c])
                return Tensor(out, (len(rows), c))
        raise IndexError("Only 1-D and 2-D indexing supported")

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={list(self.data[:6])}{'...' if len(self.data) > 6 else ''})"

    # ------------------------------------------------------------------ #
    # Precision casting
    # ------------------------------------------------------------------ #
    def to_int8(self, scale, zero_point):
        out = array.array("b", [0] * len(self.data))
        for i, v in enumerate(self.data):
            q = int(round(v / scale)) + zero_point
            out[i] = max(INT8_MIN, min(INT8_MAX, q))
        return out

    def to_float32(self):
        return Tensor(list(self.data), self.shape)

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def tolist(self):
        return list(self.data)

    def copy(self):
        return Tensor(array.array("f", self.data), self.shape)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _prod(shape):
    p = 1
    for s in shape:
        p *= s
    return p
