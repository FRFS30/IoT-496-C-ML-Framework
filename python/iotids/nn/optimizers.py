import math

try:
    import numpy as np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False


def _clip_norm(grads, max_norm):
    """Global gradient norm clipping — important for FL training stability."""
    if _HAS_NP:
        total_sq = sum(float(np.sum(g ** 2)) for g in grads)
        norm = math.sqrt(total_sq)
        if norm > max_norm and norm > 0:
            scale = max_norm / norm
            return [g * scale for g in grads]
        return grads
    total_sq = 0.0
    for g in grads:
        if isinstance(g, list):
            for v in g:
                total_sq += v * v
        else:
            total_sq += g * g
    norm = math.sqrt(total_sq)
    if norm > max_norm and norm > 0:
        scale = max_norm / norm
        clipped = []
        for g in grads:
            if isinstance(g, list):
                clipped.append([v * scale for v in g])
            else:
                clipped.append(g * scale)
        return clipped
    return grads


class Adam:
    """Adam with bias correction. NumPy-native when available."""

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, clip_norm=None):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.clip_norm = clip_norm
        self._t = 0
        self._m = {}
        self._v = {}

    def step(self, layers):
        self._t += 1
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t
        lr_t = self.lr * math.sqrt(bc2) / bc1

        for layer in layers:
            for pid, (param, grad) in enumerate(_get_param_grad_pairs(layer)):
                key = (id(layer), pid)

                if _HAS_NP:
                    g = grad.copy()
                    if self.clip_norm:
                        g = _clip_norm([g], self.clip_norm)[0]
                    if key not in self._m:
                        self._m[key] = np.zeros_like(param)
                        self._v[key] = np.zeros_like(param)
                    m, v = self._m[key], self._v[key]
                    m *= self.beta1;  m += (1.0 - self.beta1) * g
                    v *= self.beta2;  v += (1.0 - self.beta2) * g * g
                    # In-place update on the actual parameter array
                    param -= lr_t * m / (np.sqrt(v) + self.eps)
                else:
                    # Pure-Python fallback
                    g_flat = list(grad) if not isinstance(grad, list) else grad
                    if self.clip_norm:
                        g_flat = _clip_norm([g_flat], self.clip_norm)[0]
                    if key not in self._m:
                        self._m[key] = [0.0] * len(param)
                        self._v[key] = [0.0] * len(param)
                    m, v = self._m[key], self._v[key]
                    for i in range(len(param)):
                        g_i = g_flat[i]
                        m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g_i
                        v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g_i * g_i
                        param[i] -= lr_t * m[i] / (math.sqrt(v[i]) + self.eps)


class SGD:
    """SGD with optional momentum. NumPy-native when available."""

    def __init__(self, lr=1e-2, momentum=0.0, clip_norm=None):
        self.lr = lr
        self.momentum = momentum
        self.clip_norm = clip_norm
        self._velocity = {}

    def step(self, layers):
        for layer in layers:
            for pid, (param, grad) in enumerate(_get_param_grad_pairs(layer)):
                key = (id(layer), pid)

                if _HAS_NP:
                    g = grad.copy()
                    if self.clip_norm:
                        g = _clip_norm([g], self.clip_norm)[0]
                    if self.momentum > 0:
                        if key not in self._velocity:
                            self._velocity[key] = np.zeros_like(param)
                        vel = self._velocity[key]
                        vel *= self.momentum;  vel += g
                        param -= self.lr * vel
                    else:
                        param -= self.lr * g
                else:
                    g_flat = list(grad)
                    if self.clip_norm:
                        g_flat = _clip_norm([g_flat], self.clip_norm)[0]
                    if self.momentum > 0:
                        if key not in self._velocity:
                            self._velocity[key] = [0.0] * len(param)
                        vel = self._velocity[key]
                        for i in range(len(param)):
                            vel[i] = self.momentum * vel[i] + g_flat[i]
                            param[i] -= self.lr * vel[i]
                    else:
                        for i in range(len(param)):
                            param[i] -= self.lr * g_flat[i]


# ------------------------------------------------------------------ #
# Internal helper — yields (param_array, grad_array) pairs per layer
# ------------------------------------------------------------------ #
def _get_param_grad_pairs(layer):
    """
    Yields (param, grad) pairs where both are the same array type
    (numpy ndarray when available, plain list otherwise).

    For Dense:  W is (in_features, units), dW same shape — yield as-is.
                b is (units,), db same — yield as-is.
    For BatchNormalization: gamma, beta as 1-D arrays.

    The optimizer receives the *actual* mutable array objects so in-place
    updates (param -= ...) write back to the layer without extra copies.
    """
    from .layers import Dense, BatchNormalization

    if isinstance(layer, Dense):
        # W and dW: numpy arrays of shape (in_features, units)
        # Yield the whole matrix as one param group — the vectorised Adam
        # update handles the full matrix in one shot.
        yield (layer.W, layer.dW)
        if layer.use_bias and layer.b is not None:
            yield (layer.b, layer.db)

    elif isinstance(layer, BatchNormalization):
        yield (layer.gamma, layer.dgamma)
        yield (layer.beta,  layer.dbeta)