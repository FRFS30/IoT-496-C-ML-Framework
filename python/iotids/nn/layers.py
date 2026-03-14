import math
import array
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
    """Fully connected layer with He / Glorot init."""

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

        self.W = [[rng.gauss(0, std) for _ in range(units)] for _ in range(in_features)]
        self.b = [0.0] * units if use_bias else None

        self.dW = [[0.0] * units for _ in range(in_features)]
        self.db = [0.0] * units if use_bias else None

        self._cache_X = None
        self._cache_Z = None

    def forward(self, X):
        """X: list of rows [[...], ...]  -> list of rows."""
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
        # Apply activation
        A = [[self._act(v) for v in z] for z in Z]
        return A

    def backward(self, grad_out):
        """grad_out: gradient w.r.t. layer output (list of rows)."""
        X = self._cache_X
        Z = self._cache_Z
        batch = len(X)

        # Gradient through activation
        if self._act_name == "sigmoid":
            # grad_out * sigmoid_deriv(A) — A = sigmoid(Z)
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

        # dW
        for i in range(self.in_features):
            for j in range(self.units):
                g = 0.0
                for b in range(batch):
                    g += X[b][i] * dZ[b][j]
                self.dW[i][j] = g / batch

        # db
        if self.use_bias:
            for j in range(self.units):
                self.db[j] = sum(dZ[b][j] for b in range(batch)) / batch

        # Gradient w.r.t. input
        grad_in = []
        for b in range(batch):
            row = [0.0] * self.in_features
            for i in range(self.in_features):
                for j in range(self.units):
                    row[i] += dZ[b][j] * self.W[i][j]
            grad_in.append(row)

        return grad_in

    def get_weights(self):
        flat_W = [v for row in self.W for v in row]
        return [flat_W, list(self.b) if self.use_bias else []]

    def set_weights(self, weights):
        flat_W, b = weights[0], weights[1]
        for i in range(self.in_features):
            for j in range(self.units):
                self.W[i][j] = flat_W[i * self.units + j]
        if self.use_bias and b:
            self.b = list(b)


# ------------------------------------------------------------------ #
# BatchNormalization  (build last in nn/ — see implementation notes)
# ------------------------------------------------------------------ #
class BatchNormalization(Layer):
    """Running mean/var, learnable gamma/beta."""

    def __init__(self, features, eps=1e-5, momentum=0.1):
        self.features = features
        self.eps = eps
        self.momentum = momentum

        self.gamma = [1.0] * features
        self.beta  = [0.0] * features
        self.running_mean = [0.0] * features
        self.running_var  = [1.0] * features

        self.dgamma = [0.0] * features
        self.dbeta  = [0.0] * features

        self._cache = None

    def forward(self, X):
        batch = len(X)
        f = self.features

        if self.training:
            # Batch mean/var
            mean = [sum(X[b][j] for b in range(batch)) / batch for j in range(f)]
            var  = [sum((X[b][j] - mean[j]) ** 2 for b in range(batch)) / batch for j in range(f)]

            # Normalise
            X_hat = [[(X[b][j] - mean[j]) / math.sqrt(var[j] + self.eps) for j in range(f)]
                     for b in range(batch)]
            out = [[self.gamma[j] * X_hat[b][j] + self.beta[j] for j in range(f)]
                   for b in range(batch)]

            # Update running stats
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

    def backward(self, grad_out):
        X, X_hat, mean, var = self._cache
        batch = len(X)
        f = self.features

        self.dgamma = [sum(grad_out[b][j] * X_hat[b][j] for b in range(batch)) for j in range(f)]
        self.dbeta  = [sum(grad_out[b][j] for b in range(batch)) for j in range(f)]

        dX_hat = [[grad_out[b][j] * self.gamma[j] for j in range(f)] for b in range(batch)]
        inv_std = [1.0 / math.sqrt(var[j] + self.eps) for j in range(f)]

        # Full backprop through batch norm
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
        return [list(self.gamma), list(self.beta),
                list(self.running_mean), list(self.running_var)]

    def set_weights(self, weights):
        self.gamma, self.beta = list(weights[0]), list(weights[1])
        self.running_mean, self.running_var = list(weights[2]), list(weights[3])


# ------------------------------------------------------------------ #
# Dropout
# ------------------------------------------------------------------ #
class Dropout(Layer):
    """Inverted dropout — training mode only."""

    def __init__(self, rate=0.5):
        self.rate = rate
        self._mask = None

    def forward(self, X):
        if not self.training:
            return X
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
        return [[g * m for g, m in zip(grad_row, mask_row)]
                for grad_row, mask_row in zip(grad_out, self._mask)]
