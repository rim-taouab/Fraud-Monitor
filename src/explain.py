"""Per-alert explanations via TreeSHAP.

Uses XGBoost's native SHAP (`pred_contribs=True`) so every alert can show its
top-3 reasons without pulling in the external `shap` package. Each contribution
is in log-odds units: positive pushes the score toward "fraud".
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb


def shap_contributions(xgb_detector, X, feature_names):
    """Return (contribs, base_value).

    contribs: (n_rows, n_features) SHAP values aligned to feature_names.
    base_value: the model's bias term (expected log-odds).
    """
    booster = xgb_detector.model.get_booster()
    dm = xgb.DMatrix(np.asarray(X), feature_names=list(feature_names))
    raw = booster.predict(dm, pred_contribs=True)  # (n, n_features + 1)
    return raw[:, :-1], raw[:, -1]


def top_reasons(contribs_row, feature_names, k: int = 3):
    """Top-k features pushing THIS transaction toward fraud (largest positive)."""
    order = np.argsort(contribs_row)[::-1]
    out = []
    for j in order[:k]:
        if contribs_row[j] <= 0:
            break
        out.append((feature_names[j], float(contribs_row[j])))
    return out
