import math
import numpy as np


def _clip_norm(grads, max_norm):
    """Global gradient norm clipping — important for FL training stability."""
    total_sq = sum(float(np.sum(g ** 2)) for g in grads)
    norm = math.sqrt(total_sq)
    if norm > max_norm and norm > 0:
        scale = max_norm / norm
        return [g * scale for g in grads]
    return grads


class Adam:
    """Adam with bias correction. NumPy-native."""

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 clip_norm=None):
        self.lr        = lr
        self.beta1     = beta1
        self.beta2     = beta2
        self.eps       = eps
        self.clip_norm = clip_norm
        self._t = 0
        self._m = {}
        self._v = {}

    def step(self, layers):
        self._t += 1
        bc1  = 1.0 - self.beta1 ** self._t
        bc2  = 1.0 - self.beta2 ** self._t
        lr_t = self.lr * math.sqrt(bc2) / bc1

        for layer in layers:
            for pid, (param, grad) in enumerate(
                    _get_param_grad_pairs(layer)):
                key = (id(layer), pid)

                g = grad.copy()
                if self.clip_norm:
                    g = _clip_norm([g], self.clip_norm)[0]

                if key not in self._m:
                    self._m[key] = np.zeros_like(param)
                    self._v[key] = np.zeros_like(param)
                m, v = self._m[key], self._v[key]

                m *= self.beta1
                m += (1.0 - self.beta1) * g
                v *= self.beta2
                v += (1.0 - self.beta2) * g * g

                param -= lr_t * m / (np.sqrt(v) + self.eps)


class SGD:
    """SGD with optional momentum. NumPy-native."""

    def __init__(self, lr=1e-2, momentum=0.0, clip_norm=None):
        self.lr        = lr
        self.momentum  = momentum
        self.clip_norm = clip_norm
        self._velocity = {}

    def step(self, layers):
        for layer in layers:
            for pid, (param, grad) in enumerate(
                    _get_param_grad_pairs(layer)):
                key = (id(layer), pid)

                g = grad.copy()
                if self.clip_norm:
                    g = _clip_norm([g], self.clip_norm)[0]

                if self.momentum > 0:
                    if key not in self._velocity:
                        self._velocity[key] = np.zeros_like(param)
                    vel = self._velocity[key]
                    vel *= self.momentum
                    vel += g
                    param -= self.lr * vel
                else:
                    param -= self.lr * g


# ------------------------------------------------------------------ #
# Internal helper
# ------------------------------------------------------------------ #
def _get_param_grad_pairs(layer):
    """
    Yields (param, grad) as numpy arrays.
    Dense  : W (in_features, units) then b (units,)
    BN     : gamma, beta
    In-place updates on param write back to the layer directly.
    """
    from .layers import Dense, BatchNormalization

    if isinstance(layer, Dense):
        yield (layer.W, layer.dW)
        if layer.use_bias and layer.b is not None:
            yield (layer.b, layer.db)

    elif isinstance(layer, BatchNormalization):
        yield (layer.gamma, layer.dgamma)
        yield (layer.beta,  layer.dbeta)