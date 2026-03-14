from ..nn.layers import Dense


def prune_neurons(layer, pct):
    """
    Remove entire output neurons (columns of W) with lowest L2 norm.
    Physically reduces layer.units — rebuilds weight matrix.

    layer: Dense
    pct  : fraction of neurons to remove (0.0 – 1.0)
    Returns list of surviving neuron indices.
    """
    units = layer.units
    n_remove = max(0, int(units * pct))
    if n_remove == 0:
        return list(range(units))

    # L2 norm per output neuron
    norms = []
    for j in range(units):
        norm_sq = sum(layer.W[i][j] ** 2 for i in range(layer.in_features))
        norms.append((norm_sq ** 0.5, j))

    norms.sort()
    remove_set = set(j for _, j in norms[:n_remove])
    keep = [j for j in range(units) if j not in remove_set]

    # Rebuild W, b, dW, db with surviving neurons only
    layer.W  = [[layer.W[i][j]  for j in keep] for i in range(layer.in_features)]
    layer.dW = [[layer.dW[i][j] for j in keep] for i in range(layer.in_features)]
    if layer.use_bias:
        layer.b  = [layer.b[j]  for j in keep]
        layer.db = [layer.db[j] for j in keep]
    layer.units = len(keep)

    return keep


def prune_heads(layer, pct):
    """
    For multi-output final layers: remove output heads with smallest norm.
    Equivalent to prune_neurons for Dense output layers.
    """
    return prune_neurons(layer, pct)


def rebuild_next_layer(prev_layer, next_layer, surviving_indices):
    """
    After pruning prev_layer's outputs, trim the corresponding input
    rows in next_layer so shapes stay consistent.

    prev_layer       : Dense (already pruned)
    next_layer       : Dense
    surviving_indices: list returned by prune_neurons
    """
    next_layer.W  = [next_layer.W[i]  for i in surviving_indices]
    next_layer.dW = [next_layer.dW[i] for i in surviving_indices]
    next_layer.in_features = len(surviving_indices)
