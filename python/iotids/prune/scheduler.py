import math
from .magnitude import magnitude_prune, get_sparsity


class PolynomialDecayScheduler:
    """
    Gradually increases sparsity from start_sparsity to end_sparsity
    over n_steps using a polynomial curve (default: power=3 cubic).
    """

    def __init__(self, model, start_sparsity=0.0, end_sparsity=0.5,
                 n_steps=10, power=3):
        self.model          = model
        self.start_sparsity = start_sparsity
        self.end_sparsity   = end_sparsity
        self.n_steps        = n_steps
        self.power          = power
        self._step          = 0
        self._log           = []   # (step, sparsity, val_acc)

    def on_epoch_end(self, epoch, logs=None):
        """Call at end of each training epoch (integrates with model.fit callbacks)."""
        self._step += 1
        target = self._compute_target(self._step)
        masks = magnitude_prune(self.model, target)
        actual = get_sparsity(self.model)
        val_acc = (logs or {}).get("val_acc", None)
        self._log.append((self._step, actual, val_acc))
        return masks

    def _compute_target(self, step):
        t = min(step / self.n_steps, 1.0)
        decay = (1.0 - t) ** self.power
        return self.end_sparsity + (self.start_sparsity - self.end_sparsity) * decay

    def get_log(self):
        return self._log


class GradualWarmupPruner:
    """
    Avoids the accuracy cliff by warming up from zero sparsity
    before handing off to PolynomialDecayScheduler.

    warmup_epochs: epochs at 0% sparsity before pruning starts
    then delegates to PolynomialDecayScheduler for the remainder.
    """

    def __init__(self, model, warmup_epochs=5, end_sparsity=0.5,
                 prune_epochs=15, power=3):
        self.warmup_epochs = warmup_epochs
        self._scheduler = PolynomialDecayScheduler(
            model,
            start_sparsity=0.0,
            end_sparsity=end_sparsity,
            n_steps=prune_epochs,
            power=power,
        )
        self._epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self._epoch += 1
        if self._epoch > self.warmup_epochs:
            return self._scheduler.on_epoch_end(epoch, logs)
        return None

    def get_log(self):
        return self._scheduler.get_log()
