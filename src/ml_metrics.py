"""Scoring for imbalanced detection. PR-AUC and recall-at-fixed-precision,
never ROC-AUC or accuracy.

Why not ROC-AUC: at 0.18% positives, ROC-AUC looks great even for a weak model
because true negatives dominate. The precision-recall curve conditions on the
alerts we actually raise, which is what an investigations team cares about.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def pr_auc(y_true, scores) -> float:
    """Area under the precision-recall curve (average precision)."""
    return float(average_precision_score(y_true, scores))


def recall_at_precision(y_true, scores, min_precision: float) -> float:
    """Highest recall achievable while precision >= min_precision.

    Returns 0.0 if no threshold reaches that precision.
    """
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ok = precision >= min_precision
    return float(np.max(recall[ok])) if ok.any() else 0.0


def precision_at_recall(y_true, scores, min_recall: float) -> float:
    """Highest precision achievable while recall >= min_recall."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ok = recall >= min_recall
    return float(np.max(precision[ok])) if ok.any() else 0.0
