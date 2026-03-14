import math
import random


def iid_partition(X, y, n_clients, seed=42):
    """
    Stratified random split — each client gets a balanced class distribution.
    Returns list of (X_i, y_i) per client.
    """
    random.seed(seed)

    # Group indices by class
    class_idx = {}
    for i, label in enumerate(y):
        class_idx.setdefault(label, []).append(i)

    # Shuffle within each class
    for idxs in class_idx.values():
        random.shuffle(idxs)

    # Distribute round-robin per class
    client_indices = [[] for _ in range(n_clients)]
    for idxs in class_idx.values():
        for j, idx in enumerate(idxs):
            client_indices[j % n_clients].append(idx)

    # Shuffle each client's local data
    for ci in client_indices:
        random.shuffle(ci)

    return [([X[i] for i in ci], [y[i] for i in ci]) for ci in client_indices]


def non_iid_partition(X, y, n_clients, alpha=0.5, seed=42):
    """
    Dirichlet distribution over class labels per client.
    alpha controls skew: small alpha (0.1) = severe non-IID,
    large alpha (10+) approaches IID.

    This is the key experiment for the conference — clients see
    different attack distributions.
    """
    random.seed(seed)

    classes = sorted(set(y))
    class_idx = {c: [] for c in classes}
    for i, label in enumerate(y):
        class_idx[label].append(i)

    for idxs in class_idx.values():
        random.shuffle(idxs)

    client_indices = [[] for _ in range(n_clients)]

    for cls in classes:
        idxs = class_idx[cls]
        # Sample Dirichlet proportions
        props = _dirichlet([alpha] * n_clients)
        # Assign data according to proportions
        cum = 0
        for c in range(n_clients):
            share = round(props[c] * len(idxs))
            client_indices[c].extend(idxs[cum: cum + share])
            cum += share
        # Assign any remaining to the last client
        if cum < len(idxs):
            client_indices[-1].extend(idxs[cum:])

    for ci in client_indices:
        random.shuffle(ci)

    return [([X[i] for i in ci], [y[i] for i in ci]) for ci in client_indices]


# ------------------------------------------------------------------ #
# Minimal Dirichlet sampler (no scipy)
# Uses Gamma(alpha, 1) / sum trick
# ------------------------------------------------------------------ #
def _dirichlet(alphas):
    """Sample from Dirichlet(alphas) using Gamma variates."""
    samples = [_gamma_sample(a) for a in alphas]
    total = sum(samples) or 1e-9
    return [s / total for s in samples]


def _gamma_sample(shape, scale=1.0):
    """
    Marsaglia & Tsang's method for Gamma(shape >= 1).
    For shape < 1, uses the boost trick.
    """
    if shape < 1.0:
        return _gamma_sample(1.0 + shape) * (random.random() ** (1.0 / shape))

    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = random.gauss(0, 1)
        v = 1.0 + c * x
        if v <= 0:
            continue
        v3 = v ** 3
        u = random.random()
        if u < 1.0 - 0.0331 * (x ** 4):
            return d * v3 * scale
        if math.log(u) < 0.5 * x * x + d * (1.0 - v3 + math.log(v3)):
            return d * v3 * scale
