"""Time-ordered train/test split.

NEVER split fraud data randomly: a random split lets the model/threshold learn
from transactions that happen AFTER the ones it is tested on, which inflates
every metric and collapses in production. We cut at a timestamp quantile so all
train rows precede all test rows.
"""
from __future__ import annotations

import pandas as pd

from schema import TIMESTAMP


def time_split(df: pd.DataFrame, test_frac: float = 0.25):
    """Split by time. Returns (train, test, cutoff_timestamp).

    All train timestamps < cutoff <= all test timestamps.
    """
    df = df.sort_values(TIMESTAMP, kind="mergesort").reset_index(drop=True)
    cutoff = df[TIMESTAMP].quantile(1 - test_frac)
    train = df[df[TIMESTAMP] < cutoff].reset_index(drop=True)
    test = df[df[TIMESTAMP] >= cutoff].reset_index(drop=True)
    return train, test, cutoff
