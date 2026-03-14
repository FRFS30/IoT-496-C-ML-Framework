import math
from .layers import Dense, BatchNormalization, Dropout, Layer
from .losses import BinaryCrossentropy
from .optimizers import Adam
from ..metrics.classification import (
    accuracy, precision, recall, f1_score, roc_auc, threshold_sweep,
)
from ..utils import io as _io


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self._best = None
        self._wait = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self._best is None or val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._wait = 0
            if self.restore_best:
                self.best_weights = model.get_weights()
            return False  # keep going
        self._wait += 1
        return self._wait >= self.patience

    def restore(self, model):
        if self.restore_best and self.best_weights is not None:
            model.set_weights(self.best_weights)


class LRScheduler:
    """Step decay: lr = lr0 * drop^floor(epoch / every)."""

    def __init__(self, optimizer, drop=0.5, every=10):
        self.opt = optimizer
        self.drop = drop
        self.every = every
        self._lr0 = optimizer.lr

    def step(self, epoch):
        self.opt.lr = self._lr0 * (self.drop ** (epoch // self.every))


class Sequential:
    """Ordered layer stack with fit / predict / evaluate / save / load."""

    def __init__(self, layers):
        self.layers = layers

    # ------------------------------------------------------------------ #
    # Training mode toggle
    # ------------------------------------------------------------------ #
    def _set_training(self, flag):
        for l in self.layers:
            l.training = flag

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #
    def _forward(self, X):
        out = X
        for l in self.layers:
            out = l.forward(out)
        return out

    # ------------------------------------------------------------------ #
    # Backward pass
    # ------------------------------------------------------------------ #
    def _backward(self, grad):
        for l in reversed(self.layers):
            grad = l.backward(grad)

    # ------------------------------------------------------------------ #
    # fit
    # ------------------------------------------------------------------ #
    def fit(self, X, y, epochs=20, batch_size=256, validation_split=0.15,
            optimizer=None, loss=None, callbacks=None, verbose=True):

        if optimizer is None:
            optimizer = Adam(lr=1e-3)
        if loss is None:
            loss = BinaryCrossentropy(from_logits=False)
        if callbacks is None:
            callbacks = []

        # Validation split
        n = len(y)
        n_val = max(1, int(n * validation_split))
        X_val, y_val = X[-n_val:], y[-n_val:]
        X_tr,  y_tr  = X[:-n_val], y[:-n_val]

        history = {"loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            self._set_training(True)

            # Shuffle training data
            indices = list(range(len(y_tr)))
            _fisher_yates(indices)
            X_tr = [X_tr[i] for i in indices]
            y_tr = [y_tr[i] for i in indices]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(y_tr), batch_size):
                end = min(start + batch_size, len(y_tr))
                Xb = X_tr[start:end]
                yb = y_tr[start:end]

                # Forward
                out = self._forward(Xb)
                preds = [row[-1] if isinstance(row, list) else row for row in out]

                # Loss
                batch_loss = loss(yb, preds)
                epoch_loss += batch_loss

                # Gradient of loss w.r.t. final layer output
                grad_loss = loss.gradient(yb, preds)
                # Shape into list-of-rows matching final layer output
                grad = [[g] for g in grad_loss]

                # Backward
                self._backward(grad)

                # Optimiser step
                optimizer.step(self.layers)
                n_batches += 1

            epoch_loss /= n_batches

            # Validation
            self._set_training(False)
            val_preds_raw = self._forward(X_val)
            val_preds = [row[-1] if isinstance(row, list) else row for row in val_preds_raw]
            val_loss = loss(y_val, val_preds)
            val_labels = [1 if p >= 0.5 else 0 for p in val_preds]
            val_acc = accuracy(y_val, val_labels)

            history["loss"].append(epoch_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | loss={epoch_loss:.4f} "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            # Callbacks (EarlyStopping, LRScheduler)
            stop = False
            for cb in callbacks:
                if isinstance(cb, EarlyStopping):
                    if cb(val_loss, self):
                        if verbose:
                            print(f"  EarlyStopping at epoch {epoch+1}")
                        cb.restore(self)
                        stop = True
                elif isinstance(cb, LRScheduler):
                    cb.step(epoch)
            if stop:
                break

        return history

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict(self, X):
        self._set_training(False)
        raw = self._forward(X)
        return [row[-1] if isinstance(row, list) else row for row in raw]

    def predict_threshold(self, X, t=0.5):
        probs = self.predict(X)
        return [1 if p >= t else 0 for p in probs]

    def evaluate(self, X, y, threshold=0.5):
        probs = self.predict(X)
        labels = [1 if p >= t else 0 for p, t in zip(probs, [threshold] * len(probs))]
        return {
            "accuracy":  accuracy(y, labels),
            "precision": precision(y, labels),
            "recall":    recall(y, labels),
            "f1":        f1_score(y, labels),
            "auc":       roc_auc(y, probs),
        }

    # ------------------------------------------------------------------ #
    # Weight access — FedAvg aggregation
    # ------------------------------------------------------------------ #
    def get_weights(self):
        return [l.get_weights() for l in self.layers]

    def set_weights(self, weights):
        for l, w in zip(self.layers, weights):
            if w:
                l.set_weights(w)

    # ------------------------------------------------------------------ #
    # Save / load — compact binary format via utils/io
    # ------------------------------------------------------------------ #
    def save(self, path):
        state = {"weights": self.get_weights()}
        _io.save(state, path)

    def load(self, path):
        state = _io.load(path)
        self.set_weights(state["weights"])


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _fisher_yates(lst):
    import random
    n = len(lst)
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        lst[i], lst[j] = lst[j], lst[i]
