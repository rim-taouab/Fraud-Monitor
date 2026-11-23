"""The Phase 2 detectors: a logistic-regression floor, an XGBoost booster, and
an unsupervised Isolation Forest. Each exposes `.fit(X, y)` and a `.score(X)`
that returns a fraud score in [0, 1]-ish where higher = more suspicious, so they
can all be dropped into the same evaluation.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class LogRegDetector:
    """Linear sanity floor. Scaled inputs + class_weight to handle imbalance."""

    name = "logreg"

    def __init__(self):
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]


class XGBDetector:
    """Gradient boosting. scale_pos_weight rebalances the rare fraud class."""

    name = "xgboost"

    def __init__(self):
        self.model = None
        self._params = dict(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
        )

    def fit(self, X, y):
        y = np.asarray(y)
        pos = max(int(y.sum()), 1)
        neg = len(y) - pos
        self.model = XGBClassifier(scale_pos_weight=neg / pos, **self._params)
        self.model.fit(X, y)
        return self

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]


class IsoForestDetector:
    """Unsupervised: learns 'normal' and flags outliers. Catches fraud TYPES
    it never saw labelled -- the point of keeping it alongside the supervised
    models. Trained on features only (labels ignored)."""

    name = "isolation_forest"

    def __init__(self):
        self.model = make_pipeline(
            StandardScaler(),
            IsolationForest(n_estimators=300, contamination="auto", random_state=0),
        )

    def fit(self, X, y=None):
        self.model.fit(X)
        return self

    def score(self, X):
        # score_samples: higher = more normal. Negate and min-max to [0,1] so
        # higher = more suspicious, matching the supervised detectors.
        raw = -self.model.named_steps["isolationforest"].score_samples(
            self.model.named_steps["standardscaler"].transform(X)
        )
        lo, hi = raw.min(), raw.max()
        return (raw - lo) / (hi - lo) if hi > lo else raw


DETECTORS = [LogRegDetector, XGBDetector, IsoForestDetector]
