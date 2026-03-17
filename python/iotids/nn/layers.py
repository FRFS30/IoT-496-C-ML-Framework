import math
import array
import numpy as np
from . import activations
from ..utils import random as rng


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
    NumPy hot path for forward/backward.
    get_weights / set_weights use plain Python lists for FedAvg compatibility.
    Pure-Python fallback methods kept for reference — never called on server.
    """

    def __init__(self, in_features, units, activation=None, use_bias=True):
        self.in_features = in_features
        self.units       = units
        self.use_bias    = use_bias
        self._act, self._act_d = activations.get(activation)
        self._act_name   = activation

        if activation == "relu":
            std = math.sqrt(2.0 / in_features)
        else:
            std = math.sqrt(2.0 / (in_features + units))

        self.W  = np.random.normal(0.0, std, (in_features, units)).astype(np.float64)
        self.b  = np.zeros(units, dtype=np.float64) if use_bias else None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if use_bias else None

        self._cache_X = None
        self._cache_Z = None

    def forward(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float64)
        Z = X @ self.W
        if self.use_bias:
            Z += self.b
        self._cache_X = X
        self._cache_Z = Z

        name = self._act_name
        if name == "relu":
            return np.maximum(0.0, Z)
        elif name == "leaky_relu":
            return np.where(Z > 0, Z, 0.01 * Z)
        elif name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))
        else:
            return Z.copy()

    def backward(self, grad_out):
        if not isinstance(grad_out, np.ndarray):
            grad_out = np.array(grad_out, dtype=np.float64)
        X     = self._cache_X
        Z     = self._cache_Z
        batch = X.shape[0]

        name = self._act_name
        if name == "relu":
            dZ = grad_out * (Z > 0).astype(np.float64)
        elif name == "leaky_relu":
            dZ = grad_out * np.where(Z > 0, 1.0, 0.01)
        elif name == "sigmoid":
            A  = 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))
            dZ = grad_out * A * (1.0 - A)
        else:
            dZ = grad_out

        self.dW = X.T @ dZ / batch
        if self.use_bias:
            self.db = dZ.sum(axis=0) / batch

        return dZ @ self.W.T

    def get_weights(self):
        return [self.W.flatten().tolist(),
                self.b.tolist() if self.use_bias else []]

    def set_weights(self, weights):
        flat_W, b = weights[0], weights[1]
        self.W = np.array(flat_W, dtype=np.float64).reshape(
            self.in_features, self.units)
        if self.use_bias and b:
            self.b = np.array(b, dtype=np.float64)


# ------------------------------------------------------------------ #
# BatchNormalization
# ------------------------------------------------------------------ #
class BatchNormalization(Layer):
    """Running mean/var, learnable gamma/beta. NumPy-accelerated."""

    def __init__(self, features, eps=1e-5, momentum=0.1):
        self.features = features
        self.eps      = eps
        self.momentum = momentum

        self.gamma        = np.ones(features,  dtype=np.float64)
        self.beta         = np.zeros(features, dtype=np.float64)
        self.running_mean = np.zeros(features, dtype=np.float64)
        self.running_var  = np.ones(features,  dtype=np.float64)
        self.dgamma       = np.zeros(features, dtype=np.float64)
        self.dbeta        = np.zeros(features, dtype=np.float64)

        self._cache = None

    def forward(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float64)
        if self.training:
            mean  = X.mean(axis=0)
            var   = X.var(axis=0)
            X_hat = (X - mean) / np.sqrt(var + self.eps)
            out   = self.gamma * X_hat + self.beta
            self.running_mean = ((1 - self.momentum) * self.running_mean
                                 + self.momentum * mean)
            self.running_var  = ((1 - self.momentum) * self.running_var
                                 + self.momentum * var)
            self._cache = (X, X_hat, mean, var)
        else:
            X_hat = ((X - self.running_mean)
                     / np.sqrt(self.running_var + self.eps))
            out   = self.gamma * X_hat + self.beta
        return out

    def backward(self, grad_out):
        if not isinstance(grad_out, np.ndarray):
            grad_out = np.array(grad_out, dtype=np.float64)
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

    def get_weights(self):
        return [self.gamma.tolist(), self.beta.tolist(),
                self.running_mean.tolist(), self.running_var.tolist()]

    def set_weights(self, weights):
        self.gamma        = np.array(weights[0], dtype=np.float64)
        self.beta         = np.array(weights[1], dtype=np.float64)
        self.running_mean = np.array(weights[2], dtype=np.float64)
        self.running_var  = np.array(weights[3], dtype=np.float64)


# ------------------------------------------------------------------ #
# Dropout
# ------------------------------------------------------------------ #
class Dropout(Layer):
    """Inverted dropout — training mode only. NumPy-accelerated."""

    def __init__(self, rate=0.5):
        self.rate  = rate
        self._mask = None

    def forward(self, X):
        if not self.training:
            return X
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float64)
        scale      = 1.0 / (1.0 - self.rate)
        self._mask = (np.random.random(X.shape) >= self.rate).astype(
            np.float64) * scale
        return X * self._mask

    def backward(self, grad_out):
        if not self.training or self._mask is None:
            return grad_out
        if not isinstance(grad_out, np.ndarray):
            grad_out = np.array(grad_out, dtype=np.float64)
        return grad_out * self._mask