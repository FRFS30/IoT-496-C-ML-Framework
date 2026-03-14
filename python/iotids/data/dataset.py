import array
import math
from ..utils import random as rng
from ..core.tensor import Tensor


class Dataset:
    """Container for (X, y) pairs with batching and splitting."""

    def __init__(self, X, y):
        """X: list of float rows, y: list of labels."""
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    # ------------------------------------------------------------------ #
    # Splitting
    # ------------------------------------------------------------------ #
    def train_test_split(self, test_size=0.15, stratify=True, seed=42):
        rng.set_seed(seed)
        n = len(self.y)
        indices = list(range(n))

        if stratify:
            # Group by class
            class_idx = {}
            for i, label in enumerate(self.y):
                class_idx.setdefault(label, []).append(i)

            train_idx, test_idx = [], []
            for label, idxs in class_idx.items():
                rng.shuffle(idxs)
                n_test = max(1, int(len(idxs) * test_size))
                test_idx.extend(idxs[:n_test])
                train_idx.extend(idxs[n_test:])
        else:
            rng.shuffle(indices)
            n_test = int(n * test_size)
            test_idx = indices[:n_test]
            train_idx = indices[n_test:]

        return (
            Dataset([self.X[i] for i in train_idx], [self.y[i] for i in train_idx]),
            Dataset([self.X[i] for i in test_idx],  [self.y[i] for i in test_idx]),
        )

    def train_val_test_split(self, val_size=0.15, test_size=0.15, stratify=True, seed=42):
        train_ds, test_ds = self.train_test_split(test_size=test_size, stratify=stratify, seed=seed)
        train_ds, val_ds = train_ds.train_test_split(test_size=val_size, stratify=stratify, seed=seed + 1)
        return train_ds, val_ds, test_ds

    # ------------------------------------------------------------------ #
    # Batching
    # ------------------------------------------------------------------ #
    def batch(self, size):
        """Yield (X_batch_tensor, y_batch_tensor) pairs."""
        n = len(self.y)
        for start in range(0, n, size):
            end = min(start + size, n)
            X_batch = self.X[start:end]
            y_batch = self.y[start:end]
            yield (
                Tensor([v for row in X_batch for v in row], (end - start, len(X_batch[0]))),
                Tensor(y_batch, (end - start,)),
            )

    # ------------------------------------------------------------------ #
    # Shuffle
    # ------------------------------------------------------------------ #
    def shuffle(self, seed=None):
        if seed is not None:
            rng.set_seed(seed)
        indices = list(range(len(self.y)))
        rng.shuffle(indices)
        self.X = [self.X[i] for i in indices]
        self.y = [self.y[i] for i in indices]

    # ------------------------------------------------------------------ #
    # Stratified sample
    # ------------------------------------------------------------------ #
    def sample(self, frac=0.1, stratify=True, seed=42):
        """Return a small Dataset for calibration use."""
        rng.set_seed(seed)
        n = len(self.y)
        if stratify:
            class_idx = {}
            for i, label in enumerate(self.y):
                class_idx.setdefault(label, []).append(i)
            chosen = []
            for label, idxs in class_idx.items():
                k = max(1, int(len(idxs) * frac))
                chosen.extend(rng.sample(idxs, min(k, len(idxs))))
        else:
            k = max(1, int(n * frac))
            chosen = rng.sample(list(range(n)), k)
        return Dataset([self.X[i] for i in chosen], [self.y[i] for i in chosen])

    # ------------------------------------------------------------------ #
    # SMOTE-style oversampling (training set only)
    # ------------------------------------------------------------------ #
    def oversample_minority(self, seed=42):
        """Duplicate minority class rows until balanced."""
        rng.set_seed(seed)
        class_idx = {}
        for i, label in enumerate(self.y):
            class_idx.setdefault(label, []).append(i)
        max_count = max(len(v) for v in class_idx.values())
        new_X, new_y = list(self.X), list(self.y)
        for label, idxs in class_idx.items():
            deficit = max_count - len(idxs)
            while deficit > 0:
                pick = rng.choice(idxs)
                new_X.append(self.X[pick])
                new_y.append(label)
                deficit -= 1
        ds = Dataset(new_X, new_y)
        ds.shuffle(seed=seed + 1)
        return ds
