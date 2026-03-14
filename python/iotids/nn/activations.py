import math


# ------------------------------------------------------------------ #
# Forward passes (scalar inputs used inside layer loops)
# ------------------------------------------------------------------ #
def relu(x):
    return x if x > 0.0 else 0.0


def relu_deriv(x):
    return 1.0 if x > 0.0 else 0.0


def leaky_relu(x, alpha=0.01):
    return x if x > 0.0 else alpha * x


def leaky_relu_deriv(x, alpha=0.01):
    return 1.0 if x > 0.0 else alpha


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def sigmoid_deriv(a):
    """a is already the sigmoid output."""
    return a * (1.0 - a)


def softmax_vec(vec):
    mx = max(vec)
    exps = [math.exp(v - mx) for v in vec]
    s = sum(exps)
    return [e / s for e in exps]


# ------------------------------------------------------------------ #
# Lookup by name — used in layer constructors
# ------------------------------------------------------------------ #
def get(name):
    table = {
        "relu":       (relu,       relu_deriv),
        "leaky_relu": (leaky_relu, leaky_relu_deriv),
        "sigmoid":    (sigmoid,    sigmoid_deriv),
        "linear":     (lambda x: x, lambda x: 1.0),
        None:         (lambda x: x, lambda x: 1.0),
    }
    if name not in table:
        raise ValueError(f"Unknown activation: {name}")
    return table[name]
