class PruneMask:
    """
    Binary float32 mask per layer.
    Persists through QAT — masks survive quantization-aware training.
    """

    def __init__(self, shape):
        self.shape = shape
        n = shape[0] * (shape[1] if len(shape) > 1 else 1)
        self.mask = [1.0] * n   # 1 = keep, 0 = pruned
        self._frozen = False

    def apply(self, weights):
        """Elementwise zero-out. weights: flat list of floats."""
        return [w * m for w, m in zip(weights, self.mask)]

    def freeze(self):
        """Prevent weight re-growth during fine-tuning / QAT."""
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    def is_frozen(self):
        return self._frozen

    def sparsity(self):
        """Return fraction of weights currently zeroed."""
        n = len(self.mask)
        zeros = sum(1 for m in self.mask if m == 0.0)
        return zeros / n if n > 0 else 0.0

    def set_mask(self, indices_to_zero):
        """Zero out specific weight indices (unless frozen)."""
        if self._frozen:
            return
        for i in indices_to_zero:
            self.mask[i] = 0.0

    def get_mask(self):
        return list(self.mask)
