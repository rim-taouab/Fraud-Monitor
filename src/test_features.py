"""Causality test for the engineered features -- the Phase 2 counterpart of
test_rules_are_causal. Hiding future rows must not change the sequence features
of earlier rows. If it does, a feature is leaking the future into training.

(merchant_risk is excluded here: it is fit on train labels by design, and its
no-leakage property is that it never sees test labels, enforced structurally in
run_phase2.py, not by this test.)
"""
from __future__ import annotations

import numpy as np

from features import SEQUENCE_FEATURES
from features import add_causal_features
from generator import GenConfig, generate


def test_sequence_features_are_causal():
    df = generate(GenConfig(n_users=400, n_days=30, seed=3))

    full = add_causal_features(df)
    cut = int(len(full) * 0.60)

    # Recompute features using only the earlier 60% of transactions.
    past_only = add_causal_features(df.sort_values("timestamp").iloc[:cut].copy())

    a = full.sort_values("timestamp").iloc[:cut][SEQUENCE_FEATURES].to_numpy(float)
    b = past_only[SEQUENCE_FEATURES].to_numpy(float)

    assert a.shape == b.shape
    # Allow tiny float noise; anything larger means a feature saw the future.
    max_diff = np.nanmax(np.abs(a - b))
    assert max_diff < 1e-6, f"feature changed by {max_diff} when future was hidden"


if __name__ == "__main__":
    test_sequence_features_are_causal()
    print("PASS test_sequence_features_are_causal")
