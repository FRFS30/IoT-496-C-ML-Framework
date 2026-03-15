import math
import random
from .decision_tree import DecisionTree


class RandomForestClassifier:
    """
    Bootstrap-aggregated decision trees.
    get_weights / set_weights expose serialised node params for FedAvg.
    """

    def __init__(self, n_estimators=100, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features="sqrt", class_weight=None,
                 random_state=None):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features   # "sqrt", "log2", int, or None
        self.class_weight      = class_weight   # None, "balanced", or dict {0: w0, 1: w1}
        self.random_state      = random_state
        self.estimators_       = []
        self.feature_importances_ = None

        if random_state is not None:
            random.seed(random_state)

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        n_samples  = len(y)
        n_features = len(X[0])
        mf = self._resolve_max_features(n_features)

        sample_weights = self._build_sample_weights(y)

        self.estimators_ = []

        for i in range(self.n_estimators):
            if self.random_state is not None:
                random.seed(self.random_state + i)

            indices = self._bootstrap_indices(n_samples, sample_weights)
            X_boot  = [X[i] for i in indices]
            y_boot  = [y[i] for i in indices]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=mf,
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)

        self.feature_importances_ = self._compute_importances(n_features)
        return self

    # ------------------------------------------------------------------ #
    # Bootstrap helpers
    # ------------------------------------------------------------------ #
    def _build_sample_weights(self, y):
        if self.class_weight is None:
            return None

        n = len(y)
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1

        if self.class_weight == "balanced":
            n_classes = len(counts)
            weights_map = {
                cls: n / (n_classes * cnt)
                for cls, cnt in counts.items()
            }
        elif isinstance(self.class_weight, dict):
            weights_map = self.class_weight
        else:
            raise ValueError(f"Unknown class_weight: {self.class_weight!r}")

        raw   = [weights_map.get(label, 1.0) for label in y]
        total = sum(raw)
        return [w / total for w in raw]

    def _bootstrap_indices(self, n_samples, sample_weights):
        if sample_weights is None:
            return [random.randint(0, n_samples - 1) for _ in range(n_samples)]
        population = list(range(n_samples))
        return random.choices(population, weights=sample_weights, k=n_samples)

    def _resolve_max_features(self, n):
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(n)))
        if self.max_features == "log2":
            return max(1, int(math.log2(n)))
        if isinstance(self.max_features, int):
            return self.max_features
        return n

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #
    def predict_proba(self, X):
        n    = len(X)
        sums = [0.0] * n
        for tree in self.estimators_:
            probs = tree.predict_proba(X)
            for i, p in enumerate(probs):
                sums[i] += p
        k = len(self.estimators_)
        return [s / k for s in sums]

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probs]

    # ------------------------------------------------------------------ #
    # Feature importances
    # ------------------------------------------------------------------ #
    def _compute_importances(self, n_features):
        totals = [0.0] * n_features
        for tree in self.estimators_:
            _accumulate_importances(tree.root, totals)
        total = sum(totals) or 1.0
        return [v / total for v in totals]

    # ------------------------------------------------------------------ #
    # FedAvg weight interface
    # ------------------------------------------------------------------ #
    def get_weights(self):
        return [tree.get_params() for tree in self.estimators_]

    def set_weights(self, weights):
        for tree, params in zip(self.estimators_, weights):
            tree.set_params(params)


def _accumulate_importances(node, totals):
    if node is None or node.feature is None:
        return
    totals[node.feature] += 1.0
    _accumulate_importances(node.left,  totals)
    _accumulate_importances(node.right, totals)