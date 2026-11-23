"""Tests for the Phase 1 pieces. Runs under pytest OR directly:

    python src/test_baseline.py

The most important test is `test_rules_are_causal`: it proves a rule's decision
for a transaction does NOT change when future transactions are removed. That is
the no-leakage guarantee the whole project depends on.
"""
from __future__ import annotations

import numpy as np

from generator import GenConfig, generate, haversine_km
from rules import apply_rules
from schema import COLUMNS, IS_FRAUD, TIMESTAMP


def test_haversine_known_distance():
    # NYC (40.71, -74.01) to LA (34.05, -118.24) is ~3936 km.
    d = haversine_km(40.71, -74.01, 34.05, -118.24)
    assert 3800 < d < 4050, d


def test_schema_and_time_order():
    df = generate(GenConfig(n_users=300, n_days=30, seed=1))
    assert list(df.columns) == COLUMNS
    assert df[TIMESTAMP].is_monotonic_increasing
    assert df[IS_FRAUD].isin([0, 1]).all()
    assert 0 < df[IS_FRAUD].mean() < 0.05  # realistic imbalance


def test_rules_are_causal():
    """Removing FUTURE rows must not change the alert on earlier rows.

    We score the full frame, then score only the first 60% of rows, and check
    the per-rule decisions agree on the shared (earlier) rows. Any disagreement
    means a rule is peeking at the future -> leakage.
    """
    df = generate(GenConfig(n_users=400, n_days=30, seed=2))
    full = apply_rules(df)

    cut = int(len(df) * 0.60)
    past_only = apply_rules(df.iloc[:cut].copy())

    cols = ["rule_velocity", "rule_impossible_travel", "rule_amount_spike", "rule_alert"]
    a = full.iloc[:cut][cols].to_numpy()
    b = past_only[cols].to_numpy()
    mismatch = int(np.sum(a != b))
    assert mismatch == 0, f"{mismatch} decisions changed when the future was hidden"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
