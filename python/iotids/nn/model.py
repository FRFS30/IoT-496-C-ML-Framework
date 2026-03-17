import math
import numpy as np

from .layers import Dense, BatchNormalization, Dropout, Layer
from .losses import BinaryCrossentropy
from .optimizers import Adam
from ..metrics.classification import (
    accuracy, precision, recall, f1_score, roc_auc, threshold_sweep,
)
from ..utils import io as _io


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, restore_best=True):
        self.patience     = patience
        self.min_delta    = min_delta
        self.restore_best = restore_best
        self._best        = None
        self._wait        = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self._best is None or val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._wait = 0
            if self.restore_best:
                self.best_weights = model.get_weights()
            return False
        self._wait += 1
        return self._wait >= self.patience

    def restore(self, model):
        if self.restore_best and self.best_weights is not None:
            model.set_weights(self.best_weights)


class LRScheduler:
    """Step decay: lr = lr0 * drop^floor(epoch / every)."""

    def __init__(self, optimizer, drop=0.5, every=10):
        self.opt   = optimizer
        self.drop  = drop
        self.every = every
        self._lr0  = optimizer.lr

    def step(self, epoch):
        self.opt.lr = self._lr0 * (self.drop ** (epoch // self.every))


class Sequential:
    """Ordered layer stack with fit / predict / evaluate / save / load."""

    def __init__(self, layers):
        self.layers = layers

    def _set_training(self, flag):
        for l in self.layers:
            l.training = flag

    def _forward(self, X):
        out = X
        for l in self.layers:
            out = l.forward(out)
        return out

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
        n     = len(y)
        n_val = max(1, int(n * validation_split))
        X_val, y_val = X[-n_val:], y[-n_val:]
        X_tr,  y_tr  = X[:-n_val], y[:-n_val]
        n_tr = len(y_tr)

        # Convert once to numpy — O(1) fancy indexing per batch
        X_tr_np  = np.array(X_tr,  dtype=np.float64)
        X_val_np = np.array(X_val, dtype=np.float64)
        idx_np   = np.arange(n_tr, dtype=np.int64)

        history = {"loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            self._set_training(True)
            np.random.shuffle(idx_np)

            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n_tr, batch_size):
                batch_idx = idx_np[start:start + batch_size]

                Xb = X_tr_np[batch_idx]
                yb = [y_tr[i] for i in batch_idx.tolist()]

                out   = self._forward(Xb)
                preds = out[:, -1].tolist()

                batch_loss  = loss(yb, preds)
                epoch_loss += batch_loss

                grad_loss = loss.gradient(yb, preds)
                grad      = np.array(grad_loss, dtype=np.float64).reshape(-1, 1)

                self._backward(grad)
                optimizer.step(self.layers)
                n_batches += 1

                if verbose and n_batches % 50 == 0:
                    print(f"  epoch {epoch+1} step {n_batches}/"
                          f"{max(1, n_tr // batch_size)}"
                          f" loss={epoch_loss / n_batches:.4f}", flush=True)

            epoch_loss /= n_batches

            # Validation
            self._set_training(False)
            val_preds = []
            for vs in range(0, len(y_val), batch_size):
                ve = min(vs + batch_size, len(y_val))
                val_preds += self._forward(X_val_np[vs:ve])[:, -1].tolist()

            val_loss   = loss(y_val, val_preds)
            val_labels = [1 if p >= 0.5 else 0 for p in val_preds]
            val_acc    = accuracy(y_val, val_labels)

            history["loss"].append(epoch_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | loss={epoch_loss:.4f} "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
                      flush=True)

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
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float64)
        return self._forward(X)[:, -1].tolist()

    def predict_threshold(self, X, t=0.5):
        return [1 if p >= t else 0 for p in self.predict(X)]

    def evaluate(self, X, y, threshold=0.5):
        probs  = self.predict(X)
        labels = [1 if p >= threshold else 0 for p in probs]
        return {
            "accuracy":  accuracy(y, labels),
            "precision": precision(y, labels),
            "recall":    recall(y, labels),
            "f1":        f1_score(y, labels),
            "auc":       roc_auc(y, probs),
        }

    # ------------------------------------------------------------------ #
    # Weight access — FedAvg
    # ------------------------------------------------------------------ #
    def get_weights(self):
        return [l.get_weights() for l in self.layers]

    def set_weights(self, weights):
        for l, w in zip(self.layers, weights):
            if w:
                l.set_weights(w)

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #
    def save(self, path):
        _io.save({"weights": self.get_weights()}, path)

    def load(self, path):
        self.set_weights(_io.load(path)["weights"])