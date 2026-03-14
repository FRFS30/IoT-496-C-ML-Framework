import math
from ..utils.math import percentile, isnan, isinf, clip


class RobustScaler:
    """Median + IQR scaling — outlier-safe, critical for CIC-IDS-2017."""

    def __init__(self):
        self.medians_ = None
        self.iqrs_ = None
        self.n_features_ = 0

    def fit(self, X):
        """X: list of rows (each row is a list of floats)."""
        if not X:
            return self
        self.n_features_ = len(X[0])
        self.medians_ = []
        self.iqrs_ = []
        for j in range(self.n_features_):
            col = [X[i][j] for i in range(len(X)) if not isnan(X[i][j]) and not isinf(X[i][j])]
            med = percentile(col, 50) if col else 0.0
            iqr = percentile(col, 75) - percentile(col, 25)
            self.medians_.append(med)
            self.iqrs_.append(max(iqr, 1e-8))
        return self

    def transform(self, X):
        out = []
        for row in X:
            out.append([(row[j] - self.medians_[j]) / self.iqrs_[j]
                        for j in range(self.n_features_)])
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_params(self):
        return {"medians": self.medians_, "iqrs": self.iqrs_}

    def set_params(self, params):
        self.medians_ = params["medians"]
        self.iqrs_ = params["iqrs"]
        self.n_features_ = len(self.medians_)


class StandardScaler:
    """Mean / std normalisation."""

    def __init__(self):
        self.means_ = None
        self.stds_ = None
        self.n_features_ = 0

    def fit(self, X):
        if not X:
            return self
        self.n_features_ = len(X[0])
        self.means_ = []
        self.stds_ = []
        for j in range(self.n_features_):
            col = [X[i][j] for i in range(len(X)) if not isnan(X[i][j])]
            n = len(col)
            m = sum(col) / n if n else 0.0
            v = sum((v - m) ** 2 for v in col) / n if n > 1 else 1.0
            self.means_.append(m)
            self.stds_.append(max(math.sqrt(v), 1e-8))
        return self

    def transform(self, X):
        out = []
        for row in X:
            out.append([(row[j] - self.means_[j]) / self.stds_[j]
                        for j in range(self.n_features_)])
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LabelEncoder:
    """Encode string / numeric labels to 0-indexed integers."""

    def __init__(self):
        self.classes_ = []
        self._map = {}

    def fit(self, y):
        self.classes_ = sorted(set(y), key=str)
        self._map = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, y):
        return [self._map[v] for v in y]

    def inverse_transform(self, y):
        return [self.classes_[i] for i in y]

    def fit_transform(self, y):
        return self.fit(y).transform(y)


# ------------------------------------------------------------------ #
# Standalone utility functions
# ------------------------------------------------------------------ #
def clip_outliers(X, low_pct=1, high_pct=99):
    """Clip each column to [low_pct, high_pct] percentiles in-place."""
    if not X:
        return X
    ncols = len(X[0])
    lows, highs = [], []
    for j in range(ncols):
        col = [X[i][j] for i in range(len(X)) if not isnan(X[i][j])]
        lows.append(percentile(col, low_pct) if col else 0.0)
        highs.append(percentile(col, high_pct) if col else 1.0)
    for row in X:
        for j in range(ncols):
            row[j] = clip(row[j], lows[j], highs[j])
    return X


def replace_inf(X):
    """Replace +/-Inf with column median in-place."""
    if not X:
        return X
    ncols = len(X[0])
    for j in range(ncols):
        col = [X[i][j] for i in range(len(X)) if not isinf(X[i][j]) and not isnan(X[i][j])]
        med = percentile(col, 50) if col else 0.0
        for row in X:
            if isinf(row[j]) or isnan(row[j]):
                row[j] = med
    return X


def drop_nan_rows(X, y=None):
    """Remove rows where all features are NaN."""
    keep = [i for i, row in enumerate(X) if not all(isnan(v) for v in row)]
    X_out = [X[i] for i in keep]
    if y is not None:
        y_out = [y[i] for i in keep]
        return X_out, y_out
    return X_out
