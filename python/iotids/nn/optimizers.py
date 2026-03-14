import math


def _clip_norm(grads, max_norm):
    """Global gradient norm clipping — important for FL training stability."""
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
    """Adam with bias correction. One state dict per param tensor."""

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, clip_norm=None):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.clip_norm = clip_norm
        self._t = 0
        self._m = {}   # first moment  keyed by param id
        self._v = {}   # second moment

    def step(self, layers):
        """Update all layers. layers: list of Layer objects."""
        self._t += 1
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t
        lr_t = self.lr * math.sqrt(bc2) / bc1

        for layer in layers:
            param_groups = _get_param_grad_pairs(layer)
            for pid, (params, grads) in enumerate(param_groups):
                key = (id(layer), pid)
                if key not in self._m:
                    self._m[key] = [0.0] * len(params)
                    self._v[key] = [0.0] * len(params)

                g_flat = grads if isinstance(grads, list) else list(grads)
                if self.clip_norm:
                    g_flat = _clip_norm([g_flat], self.clip_norm)[0]

                m, v = self._m[key], self._v[key]
                for i in range(len(params)):
                    g = g_flat[i]
                    m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g
                    v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g * g
                    params[i] -= lr_t * m[i] / (math.sqrt(v[i]) + self.eps)


class SGD:
    """SGD with optional momentum."""

    def __init__(self, lr=1e-2, momentum=0.0, clip_norm=None):
        self.lr = lr
        self.momentum = momentum
        self.clip_norm = clip_norm
        self._velocity = {}

    def step(self, layers):
        for layer in layers:
            param_groups = _get_param_grad_pairs(layer)
            for pid, (params, grads) in enumerate(param_groups):
                key = (id(layer), pid)
                g_flat = list(grads)
                if self.clip_norm:
                    g_flat = _clip_norm([g_flat], self.clip_norm)[0]

                if self.momentum > 0:
                    if key not in self._velocity:
                        self._velocity[key] = [0.0] * len(params)
                    vel = self._velocity[key]
                    for i in range(len(params)):
                        vel[i] = self.momentum * vel[i] + g_flat[i]
                        params[i] -= self.lr * vel[i]
                else:
                    for i in range(len(params)):
                        params[i] -= self.lr * g_flat[i]


# ------------------------------------------------------------------ #
# Internal helper — extract mutable param/grad flat lists from a layer
# ------------------------------------------------------------------ #
def _get_param_grad_pairs(layer):
    from .layers import Dense, BatchNormalization
    pairs = []
    if isinstance(layer, Dense):
        # W — flatten row-major in-place references
        flat_W = [layer.W[i] for i in range(layer.in_features)]
        flat_dW = [layer.dW[i] for i in range(layer.in_features)]
        for i in range(layer.in_features):
            pairs.append((layer.W[i], layer.dW[i]))
        if layer.use_bias:
            pairs.append((layer.b, layer.db))
    elif isinstance(layer, BatchNormalization):
        pairs.append((layer.gamma, layer.dgamma))
        pairs.append((layer.beta,  layer.dbeta))
    return pairs
