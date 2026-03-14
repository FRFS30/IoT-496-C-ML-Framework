from .mask import PruneMask
from ..nn.layers import Dense


def magnitude_prune(model, sparsity, layerwise=False):
    """
    Zero out weights with smallest absolute magnitude.

    model    : Sequential
    sparsity : target fraction to prune (0.0 – 1.0)
    layerwise: if True, apply sparsity per-layer independently;
               if False, use global threshold across all weights.

    Returns dict {layer_index: PruneMask}.
    """
    dense_layers = [(i, l) for i, l in enumerate(model.layers) if isinstance(l, Dense)]

    if layerwise:
        masks = {}
        for idx, layer in dense_layers:
            flat_W = [v for row in layer.W for v in row]
            mask = _make_mask_for(flat_W, sparsity, layer.in_features, layer.units)
            masks[idx] = mask
            _apply_mask_to_dense(layer, mask)
        return masks
    else:
        # Collect all weights globally
        all_weights = []
        layer_slices = []
        for idx, layer in dense_layers:
            flat_W = [v for row in layer.W for v in row]
            layer_slices.append((idx, layer, len(all_weights), len(flat_W)))
            all_weights.extend(flat_W)

        threshold = _magnitude_threshold(all_weights, sparsity)

        masks = {}
        for idx, layer, start, length in layer_slices:
            chunk = all_weights[start: start + length]
            mask = PruneMask((layer.in_features, layer.units))
            zero_indices = [i for i, w in enumerate(chunk) if abs(w) <= threshold]
            mask.set_mask(zero_indices)
            masks[idx] = mask
            _apply_mask_to_dense(layer, mask)
        return masks


def get_sparsity(model):
    """Return current global sparsity fraction across Dense layers."""
    total, zeros = 0, 0
    for l in model.layers:
        if isinstance(l, Dense):
            for row in l.W:
                for v in row:
                    total += 1
                    if v == 0.0:
                        zeros += 1
    return zeros / total if total > 0 else 0.0


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _magnitude_threshold(weights, sparsity):
    """Return the value below which weights should be zeroed."""
    sorted_abs = sorted(abs(w) for w in weights)
    idx = int(len(sorted_abs) * sparsity)
    idx = min(idx, len(sorted_abs) - 1)
    return sorted_abs[idx]


def _make_mask_for(flat_W, sparsity, in_f, units):
    threshold = _magnitude_threshold(flat_W, sparsity)
    mask = PruneMask((in_f, units))
    zero_indices = [i for i, w in enumerate(flat_W) if abs(w) <= threshold]
    mask.set_mask(zero_indices)
    return mask


def _apply_mask_to_dense(layer, mask):
    flat_W = [v for row in layer.W for v in row]
    masked = mask.apply(flat_W)
    for i in range(layer.in_features):
        for j in range(layer.units):
            layer.W[i][j] = masked[i * layer.units + j]
