import math
import random
import multiprocessing as mp
from .decision_tree import DecisionTree


def _train_one_tree(args):
    """
    Module-level function required for multiprocessing pickling.
    Trains a single DecisionTree on a bootstrap sample and returns its
    serialised params so the result can be sent back across process boundaries.
    """
    (X_boot, y_boot, max_depth, min_samples_split,
     min_samples_leaf, mf, seed) = args

    random.seed(seed)
    tree = DecisionTree(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=mf,
    )
    tree.fit(X_boot, y_boot)
    return tree.get_params()


class RandomForestClassifier:
    """
    Bootstrap-aggregated ensemble of DecisionTrees.

    Key features
    ------------
    - class_weight : None | "balanced" | dict {0: w0, 1: w1}
        Weighted bootstrap sampling so minority class (attacks) is
        represented proportionally in every tree.
    - random_state : int | None
        Seeds Python's random module for reproducible results.
        Each tree gets seed = random_state + tree_index so trees differ.
    - n_jobs : int
        Number of parallel worker processes for tree training.
        -1 (default) uses all available CPU cores via multiprocessing.Pool.
        Pure Python stdlib — no external libraries.
    - get_weights / set_weights expose serialised node params for FedAvg
        aggregation in the federated learning pipeline.
    """

    def __init__(self, n_estimators=100, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features="sqrt", class_weight=None,
                 random_state=None, n_jobs=-1):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features   # "sqrt" | "log2" | int | None
        self.class_weight      = class_weight   # None | "balanced" | dict
        self.random_state      = random_state
        self.n_jobs            = n_jobs         # -1 = all cores
        self.estimators_       = []
        self.feature_importances_ = None

        if random_state is not None:
            random.seed(random_state)

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        """
        Train the forest.

        Steps
        -----
        1. Compute per-sample bootstrap weights from class_weight.
        2. Draw n_estimators bootstrap samples in the main process.
        3. Dispatch each (bootstrap_sample, hyperparams, seed) tuple to a
           worker process via multiprocessing.Pool.map.
        4. Workers return serialised tree params; reconstruct DecisionTree
           objects from those params.
        5. Compute mean-impurity-decrease feature importances.
        """
        n_samples  = len(y)
        n_features = len(X[0])
        mf         = self._resolve_max_features(n_features)
        sample_weights = self._build_sample_weights(y)

        # ── Build bootstrap tasks in the main process ─────────────────────
        tasks = []
        for i in range(self.n_estimators):
            seed = (self.random_state + i) if self.random_state is not None \
                   else random.randint(0, 2 ** 31)
            random.seed(seed)
            indices = self._bootstrap_indices(n_samples, sample_weights)
            X_boot  = [X[i] for i in indices]
            y_boot  = [y[i] for i in indices]
            tasks.append((X_boot, y_boot, self.max_depth,
                          self.min_samples_split, self.min_samples_leaf,
                          mf, seed))

        # ── Parallel training via stdlib multiprocessing ──────────────────
        n_cores = mp.cpu_count()
        workers = n_cores if self.n_jobs == -1 \
                  else min(self.n_jobs, n_cores)
        print(f"  Parallel training: {self.n_estimators} trees "
              f"across {workers}/{n_cores} cores...")

        with mp.Pool(processes=workers) as pool:
            all_params = pool.map(_train_one_tree, tasks)

        # ── Reconstruct DecisionTree objects from returned params ─────────
        self.estimators_ = []
        for params in all_params:
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=mf,
            )
            tree.set_params(params)
            self.estimators_.append(tree)

        self.feature_importances_ = self._compute_importances(n_features)
        return self

    # ------------------------------------------------------------------ #
    # Bootstrap helpers
    # ------------------------------------------------------------------ #
    def _build_sample_weights(self, y):
        """
        Compute a normalised per-sample probability distribution for
        weighted bootstrap sampling.

        class_weight=None      -> uniform (standard bagging)
        class_weight="balanced"-> weight inversely proportional to frequency:
                                  w_c = n / (n_classes * count_c)
        class_weight=dict      -> explicit {class: weight} mapping
        """
        if self.class_weight is None:
            return None

        n = len(y)
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1

        if self.class_weight == "balanced":
            n_classes   = len(counts)
            weights_map = {
                cls: n / (n_classes * cnt)
                for cls, cnt in counts.items()
            }
        elif isinstance(self.class_weight, dict):
            weights_map = self.class_weight
        else:
            raise ValueError(
                f"Unknown class_weight value: {self.class_weight!r}. "
                f"Expected None, 'balanced', or a dict."
            )

        raw   = [weights_map.get(label, 1.0) for label in y]
        total = sum(raw)
        return [w / total for w in raw]   # normalised probability distribution

    def _bootstrap_indices(self, n_samples, sample_weights):
        """
        Draw n_samples indices with replacement.
        Uses weighted sampling when sample_weights is provided (minority
        oversampling during bootstrap), plain uniform sampling otherwise.
        random.choices is stdlib and handles weighted sampling natively.
        """
        if sample_weights is None:
            return [random.randint(0, n_samples - 1)
                    for _ in range(n_samples)]
        return random.choices(range(n_samples),
                              weights=sample_weights, k=n_samples)

    def _resolve_max_features(self, n):
        """Resolve max_features string aliases to an integer count."""
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(n)))
        if self.max_features == "log2":
            return max(1, int(math.log2(n)))
        if isinstance(self.max_features, int):
            return self.max_features
        return n   # None -> use all features

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #
    def predict_proba(self, X):
        """
        Soft-vote ensemble: average P(class=1) across all trees.
        Returns a flat list of floats, one per sample.
        """
        n    = len(X)
        sums = [0.0] * n
        for tree in self.estimators_:
            probs = tree.predict_proba(X)
            for i, p in enumerate(probs):
                sums[i] += p
        k = len(self.estimators_)
        return [s / k for s in sums]

    def predict(self, X, threshold=0.5):
        """Hard predictions at a given decision threshold."""
        return [1 if p >= threshold else 0
                for p in self.predict_proba(X)]

    # ------------------------------------------------------------------ #
    # Feature importances  (mean impurity decrease)
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
        """
        Return serialised tree parameters for FedAvg aggregation.
        Called by federated/client.py before sending an update to the server.
        """
        return [tree.get_params() for tree in self.estimators_]

    def set_weights(self, weights):
        """
        Load aggregated tree parameters received from the FedAvg server.
        Called by federated/client.py after receiving a global model update.
        """
        for tree, params in zip(self.estimators_, weights):
            tree.set_params(params)


# ------------------------------------------------------------------ #
# Module-level helper (must be at module scope for multiprocessing)
# ------------------------------------------------------------------ #
def _accumulate_importances(node, totals):
    """Recursively accumulate split-feature counts for importance scoring."""
    if node is None or node.feature is None:
        return
    totals[node.feature] += 1.0
    _accumulate_importances(node.left,  totals)
    _accumulate_importances(node.right, totals)