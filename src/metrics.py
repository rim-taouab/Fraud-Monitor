"""Classification metrics for a binary alert vs. ground truth.

Deliberately NO accuracy: at 0.18% fraud, a detector that alerts on nothing is
99.8% accurate and useless. Precision, recall, F1 and the confusion matrix are
the honest view. We also break recall down per fraud pattern, so we can see
which attack types the rules miss (the ones the ML model must rescue).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def confusion(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def prf(y_true, y_pred) -> dict:
    c = confusion(y_true, y_pred)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, **c}


def recall_by_pattern(df_patterns: pd.Series, y_pred) -> pd.DataFrame:
    """Recall per fraud_pattern tag (blank = legit, excluded)."""
    y_pred = np.asarray(y_pred).astype(bool)
    rows = []
    for pat, mask in df_patterns.groupby(df_patterns):
        if pat == "":
            continue
        idx = mask.index
        caught = int(y_pred[df_patterns.index.get_indexer(idx)].sum())
        total = len(idx)
        rows.append({"pattern": pat, "caught": caught, "total": total,
                     "recall": caught / total if total else 0.0})
    return pd.DataFrame(rows).sort_values("pattern").reset_index(drop=True)
