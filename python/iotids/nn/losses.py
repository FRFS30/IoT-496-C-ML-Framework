import math


# ------------------------------------------------------------------ #
# Binary Cross-Entropy
# ------------------------------------------------------------------ #
class BinaryCrossentropy:
    """
    BCE loss. Supports from_logits=False (sigmoid pre-applied)
    or from_logits=True (applies sigmoid internally).
    """

    def __init__(self, from_logits=False):
        self.from_logits = from_logits
        self._cache = None

    def _sigmoid(self, v):
        if v >= 0:
            return 1.0 / (1.0 + math.exp(-v))
        e = math.exp(v)
        return e / (1.0 + e)

    def __call__(self, y_true, y_pred):
        """
        y_true: list of 0/1 floats
        y_pred: list of floats (logits or probabilities)
        Returns scalar loss.
        """
        eps = 1e-9
        n = len(y_true)
        total = 0.0
        preds = []
        for yt, yp in zip(y_true, y_pred):
            p = self._sigmoid(yp) if self.from_logits else yp
            p = max(eps, min(1.0 - eps, p))
            preds.append(p)
            total -= yt * math.log(p) + (1.0 - yt) * math.log(1.0 - p)
        self._cache = (y_true, preds)
        return total / n

    def gradient(self, y_true, y_pred):
        """
        Returns d_loss/d_output for each sample.
        Shape matches y_pred (list of floats — one per sample in batch).
        """
        eps = 1e-9
        n = len(y_true)
        grads = []
        for yt, yp in zip(y_true, y_pred):
            if self.from_logits:
                p = self._sigmoid(yp)
                # Combined sigmoid + BCE gradient = (p - y)
                grads.append((p - yt) / n)
            else:
                p = max(eps, min(1.0 - eps, yp))
                grads.append((-yt / p + (1.0 - yt) / (1.0 - p)) / n)
        return grads


# ------------------------------------------------------------------ #
# Focal Loss  (handles severe class imbalance in CIC-IDS-2017)
# ------------------------------------------------------------------ #
class FocalLoss:
    """
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    alpha: weight for positive class
    gamma: focusing exponent — higher = more focus on hard examples
    """

    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False):
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def _sigmoid(self, v):
        if v >= 0:
            return 1.0 / (1.0 + math.exp(-v))
        e = math.exp(v)
        return e / (1.0 + e)

    def __call__(self, y_true, y_pred):
        eps = 1e-9
        n = len(y_true)
        total = 0.0
        for yt, yp in zip(y_true, y_pred):
            p = self._sigmoid(yp) if self.from_logits else yp
            p = max(eps, min(1.0 - eps, p))
            p_t = p if yt == 1 else (1.0 - p)
            alpha_t = self.alpha if yt == 1 else (1.0 - self.alpha)
            total += -alpha_t * ((1.0 - p_t) ** self.gamma) * math.log(p_t)
        return total / n

    def gradient(self, y_true, y_pred):
        """Analytical gradient of focal loss w.r.t. model output."""
        eps = 1e-9
        n = len(y_true)
        grads = []
        for yt, yp in zip(y_true, y_pred):
            p = self._sigmoid(yp) if self.from_logits else yp
            p = max(eps, min(1.0 - eps, p))
            p_t = p if yt == 1 else (1.0 - p)
            alpha_t = self.alpha if yt == 1 else (1.0 - self.alpha)
            sign = 1.0 if yt == 1 else -1.0

            # d/dp_t of focal term
            mod = (1.0 - p_t) ** self.gamma
            d_log = -1.0 / p_t
            d_mod = -self.gamma * (1.0 - p_t) ** (self.gamma - 1.0)
            d_fl  = alpha_t * (d_mod * math.log(p_t) + mod * d_log)

            # chain rule through sigmoid if needed
            if self.from_logits:
                dsig = p * (1.0 - p)
                grads.append(sign * d_fl * dsig / n)
            else:
                grads.append(sign * d_fl / n)
        return grads
