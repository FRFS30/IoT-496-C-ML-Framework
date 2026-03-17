import math
import array
from . import activations
from ..utils import random as rng

# ---------------------------------------------------------------------------
# NumPy is used for the matrix-multiply hot path in Dense forward/backward.
# The rest of the library remains pure Python so the same source tree is
# importable in environments where NumPy is absent (e.g. MicroPython stubs).
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False


class Layer:
    """Base class — all layers expose get_weights / set_weights for FedAvg."""

    training = True

    def forward(self, X):
        raise NotImplementedError

    def backward(self, grad_out):
        raise NotImplementedError

    def get_weights(self):
        return []

    def set_weights(self, weights):
        pass


# ------------------------------------------------------------------ #
# Dense
# ------------------------------------------------------------------ #
class Dense(Layer):
    """
    Fully connected layer with He / Glorot init.

    Hot path (forward + backward) uses NumPy when available.
    Internally weights are stored as numpy arrays so the optimizer can
    treat them as flat 1-D views without extra copies.

    get_weights / set_weights still return plain Python lists for
    compatibility with the FedAvg aggregator and the binary save/load.
    """

    def __init__(self, in_features, units, activation=None, use_bias=True):
        self.in_features = in_features
        self.units = units
        self.use_bias = use_bias
        self._act, self._act_d = activations.get(activation)
        self._act_name = activation

        # He init for relu, Glorot otherwise
        if activation == "relu":
            std = math.sqrt(2.0 / in_features)
        else:
            std = math.sqrt(2.0 / (in_features + units))

        if _HAS_NP:
            # shape (in_features, units) — column-major friendly for matmul
            self.W  = np.random.normal(0.0, std, (in_features, units)).astype(np.float64)
            self.b  = np.zeros(units, dtype=np.float64) if use_bias else None
            self.dW = np.zeros_like(self.W)
            self.db = np.zeros_like(self.b) if use_bias else None
        else:
            self.W  = [[rng.gauss(0, std) for _ in range(units)] for _ in range(in_features)]
            self.b  = [0.0] * units if use_bias else None
            self.dW = [[0.0] * units for _ in range(in_features)]
            self.db = [0.0] * units if use_bias else None

        self._cache_X = None
        self._cache_Z = None

    # ---------------------------------------------------------------- #
    # NumPy path
    # ---------------------------------------------------------------- #
    def _forward_np(self, X):
        # X: (batch, in_features)  W: (in_features, units)  b: (units,)
        Z = X @ self.W
        if self.use_bias:
            Z += self.b
        self._cache_X = X
        self._cache_Z = Z

        # Activation — vectorized
        name = self._act_name
        if name == "relu":
            A = np.maximum(0.0, Z)
        elif name == "leaky_relu":
            A = np.where(Z > 0, Z, 0.01 * Z)
        elif name == "sigmoid":
            A = 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))
        else:
            A = Z.copy()
        return A

    def _backward_np(self, grad_out):
        # grad_out: (batch, units)
        X = self._cache_X   # (batch, in_features)
        Z = self._cache_Z   # (batch, units)
        batch = X.shape[0]

        # Gradient through activation
        name = self._act_name
        if name == "relu":
            dZ = grad_out * (Z > 0).astype(np.float64)
        elif name == "leaky_relu":
            dZ = grad_out * np.where(Z > 0, 1.0, 0.01)
        elif name == "sigmoid":
            A = 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))
            dZ = grad_out * A * (1.0 - A)
        else:
            dZ = grad_out

        # dW: (in_features, units),  db: (units,)
        self.dW = X.T @ dZ / batch
        if self.use_bias:
            self.db = dZ.sum(axis=0) / batch

        # Gradient w.r.t. input: (batch, in_features)
        grad_in = dZ @ self.W.T
        return grad_in

    # ---------------------------------------------------------------- #
    # Pure-Python path (fallback / Pico reference)
    # ---------------------------------------------------------------- #
    def _forward_py(self, X):
        self._cache_X = X
        batch = len(X)
        Z = []
        for row in X:
            z = list(self.b) if self.use_bias else [0.0] * self.units
            for j in range(self.units):
                for i in range(self.in_features):
                    z[j] += row[i] * self.W[i][j]
            Z.append(z)
        self._cache_Z = Z
        A = [[self._act(v) for v in z] for z in Z]
        return A

    def _backward_py(self, grad_out):
        X = self._cache_X
        Z = self._cache_Z
        batch = len(X)

        if self._act_name == "sigmoid":
            dZ = [[grad_out[i][j] * activations.sigmoid_deriv(activations.sigmoid(Z[i][j]))
                   for j in range(self.units)] for i in range(batch)]
        elif self._act_name == "relu":
            dZ = [[grad_out[i][j] * activations.relu_deriv(Z[i][j])
                   for j in range(self.units)] for i in range(batch)]
        elif self._act_name == "leaky_relu":
            dZ = [[grad_out[i][j] * activations.leaky_relu_deriv(Z[i][j])
                   for j in range(self.units)] for i in range(batch)]
        else:
            dZ = grad_out

        for i in range(self.in_features):
            for j in range(self.units):
                g = 0.0
                for b in range(batch):
                    g += X[b][i] * dZ[b][j]
                self.dW[i][j] = g / batch

        if self.use_bias:
            for j in range(self.units):
                self.db[j] = sum(dZ[b][j] for b in range(batch)) / batch

        grad_in = []
        for b in range(batch):
            row = [0.0] * self.in_features
            for i in range(self.in_features):
                for j in range(self.units):
                    row[i] += dZ[b][j] * self.W[i][j]
            grad_in.append(row)

        return grad_in

    # ---------------------------------------------------------------- #
    # Public interface — dispatches to NumPy or pure-Python
    # ---------------------------------------------------------------- #
    def forward(self, X):
        if _HAS_NP:
            # Accept either numpy array or list-of-lists
            if not isinstance(X, type(self.W)):  # i.e. not ndarray
                X = np.array(X, dtype=np.float64)
            return self._forward_np(X)
        return self._forward_py(X)

    def backward(self, grad_out):
        if _HAS_NP:
            if not isinstance(grad_out, type(self.W)):
                grad_out = np.array(grad_out, dtype=np.float64)
            return self._backward_np(grad_out)
        return self._backward_py(grad_out)

    # ---------------------------------------------------------------- #
    # Weight access — FedAvg / save / load
    # ---------------------------------------------------------------- #
    def get_weights(self):
        if _HAS_NP:
            return [self.W.flatten().tolist(), self.b.tolist() if self.use_bias else []]
        flat_W = [v for row in self.W for v in row]
        return [flat_W, list(self.b) if self.use_bias else []]

    def set_weights(self, weights):
        flat_W, b = weights[0], weights[1]
        if _HAS_NP:
            self.W = np.array(flat_W, dtype=np.float64).reshape(self.in_features, self.units)
            if self.use_bias and b:
                self.b = np.array(b, dtype=np.float64)
        else:
            for i in range(self.in_features):
                for j in range(self.units):
                    self.W[i][j] = flat_W[i * self.units + j]
            if self.use_bias and b:
                self.b = list(b)


# ------------------------------------------------------------------ #
# BatchNormalization
# ------------------------------------------------------------------ #
class BatchNormalization(Layer):
    """Running mean/var, learnable gamma/beta. NumPy-accelerated."""

    def __init__(self, features, eps=1e-5, momentum=0.1):
        self.features = features
        self.eps = eps
        self.momentum = momentum

        if _HAS_NP:
            self.gamma        = np.ones(features,  dtype=np.float64)
            self.beta         = np.zeros(features, dtype=np.float64)
            self.running_mean = np.zeros(features, dtype=np.float64)
            self.running_var  = np.ones(features,  dtype=np.float64)
            self.dgamma       = np.zeros(features, dtype=np.float64)
            self.dbeta        = np.zeros(features, dtype=np.float64)
        else:
            self.gamma        = [1.0] * features
            self.beta         = [0.0] * features
            self.running_mean = [0.0] * features
            self.running_var  = [1.0] * features
            self.dgamma       = [0.0] * features
            self.dbeta        = [0.0] * features

        self._cache = None

    def forward(self, X):
        if _HAS_NP:
            if not isinstance(X, np.ndarray):
                X = np.array(X, dtype=np.float64)
            return self._forward_np(X)
        return self._forward_py(X)

    def _forward_np(self, X):
        # X: (batch, features)
        if self.training:
            mean = X.mean(axis=0)
            var  = X.var(axis=0)
            X_hat = (X - mean) / np.sqrt(var + self.eps)
            out = self.gamma * X_hat + self.beta
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
            self._cache = (X, X_hat, mean, var)
        else:
            X_hat = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out   = self.gamma * X_hat + self.beta
        return out

    def backward(self, grad_out):
        if _HAS_NP:
            if not isinstance(grad_out, np.ndarray):
                grad_out = np.array(grad_out, dtype=np.float64)
            return self._backward_np(grad_out)
        return self._backward_py(grad_out)

    def _backward_np(self, grad_out):
        X, X_hat, mean, var = self._cache
        batch = X.shape[0]

        self.dgamma = (grad_out * X_hat).sum(axis=0)
        self.dbeta  = grad_out.sum(axis=0)

        dX_hat  = grad_out * self.gamma
        inv_std = 1.0 / np.sqrt(var + self.eps)

        grad_in = (1.0 / batch) * inv_std * (
            batch * dX_hat
            - dX_hat.sum(axis=0)
            - X_hat * (dX_hat * X_hat).sum(axis=0)
        )
        return grad_in

    # -- pure-Python fallbacks ------------------------------------------
    def _forward_py(self, X):
        batch = len(X)
        f = self.features
        if self.training:
            mean = [sum(X[b][j] for b in range(batch)) / batch for j in range(f)]
            var  = [sum((X[b][j] - mean[j]) ** 2 for b in range(batch)) / batch for j in range(f)]
            X_hat = [[(X[b][j] - mean[j]) / math.sqrt(var[j] + self.eps) for j in range(f)]
                     for b in range(batch)]
            out = [[self.gamma[j] * X_hat[b][j] + self.beta[j] for j in range(f)]
                   for b in range(batch)]
            for j in range(f):
                self.running_mean[j] = (1 - self.momentum) * self.running_mean[j] + self.momentum * mean[j]
                self.running_var[j]  = (1 - self.momentum) * self.running_var[j]  + self.momentum * var[j]
            self._cache = (X, X_hat, mean, var)
        else:
            X_hat = [[(X[b][j] - self.running_mean[j]) / math.sqrt(self.running_var[j] + self.eps)
                      for j in range(f)] for b in range(batch)]
            out = [[self.gamma[j] * X_hat[b][j] + self.beta[j] for j in range(f)]
                   for b in range(batch)]
        return out

    def _backward_py(self, grad_out):
        X, X_hat, mean, var = self._cache
        batch = len(X)
        f = self.features
        self.dgamma = [sum(grad_out[b][j] * X_hat[b][j] for b in range(batch)) for j in range(f)]
        self.dbeta  = [sum(grad_out[b][j] for b in range(batch)) for j in range(f)]
        dX_hat = [[grad_out[b][j] * self.gamma[j] for j in range(f)] for b in range(batch)]
        inv_std = [1.0 / math.sqrt(var[j] + self.eps) for j in range(f)]
        grad_in = []
        for b in range(batch):
            row = []
            for j in range(f):
                dxh = dX_hat[b][j]
                s = (dxh - self.dbeta[j] / batch
                     - X_hat[b][j] * self.dgamma[j] / batch)
                row.append(inv_std[j] * s)
            grad_in.append(row)
        return grad_in

    def get_weights(self):
        if _HAS_NP:
            return [self.gamma.tolist(), self.beta.tolist(),
                    self.running_mean.tolist(), self.running_var.tolist()]
        return [list(self.gamma), list(self.beta),
                list(self.running_mean), list(self.running_var)]

    def set_weights(self, weights):
        if _HAS_NP:
            self.gamma        = np.array(weights[0], dtype=np.float64)
            self.beta         = np.array(weights[1], dtype=np.float64)
            self.running_mean = np.array(weights[2], dtype=np.float64)
            self.running_var  = np.array(weights[3], dtype=np.float64)
        else:
            self.gamma, self.beta = list(weights[0]), list(weights[1])
            self.running_mean, self.running_var = list(weights[2]), list(weights[3])


# ------------------------------------------------------------------ #
# Dropout
# ------------------------------------------------------------------ #
class Dropout(Layer):
    """Inverted dropout — training mode only. NumPy-accelerated."""

    def __init__(self, rate=0.5):
        self.rate = rate
        self._mask = None

    def forward(self, X):
        if not self.training:
            return X
        if _HAS_NP:
            if not isinstance(X, np.ndarray):
                X = np.array(X, dtype=np.float64)
            scale = 1.0 / (1.0 - self.rate)
            self._mask = (np.random.random(X.shape) >= self.rate).astype(np.float64) * scale
            return X * self._mask
        # Pure-Python fallback
        scale = 1.0 / (1.0 - self.rate)
        self._mask = []
        out = []
        for row in X:
            m = [0.0 if rng.random() < self.rate else scale for _ in row]
            self._mask.append(m)
            out.append([v * d for v, d in zip(row, m)])
        return out

    def backward(self, grad_out):
        if not self.training or self._mask is None:
            return grad_out
        if _HAS_NP:
            if not isinstance(grad_out, np.ndarray):
                grad_out = np.array(grad_out, dtype=np.float64)
            return grad_out * self._mask
        return [[g * m for g, m in zip(grad_row, mask_row)]
                for grad_row, mask_row in zip(grad_out, self._mask)]