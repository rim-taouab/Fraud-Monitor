"""Tests for the cost decision layer. Checks the cost accounting against a
hand-computed example and the basic monotonic behaviour of the sweep.

    python src/test_costs.py
"""
from __future__ import annotations

import numpy as np

from costs import CostParams, expected_cost, optimal_threshold, sweep


def test_expected_cost_hand_example():
    cp = CostParams(review_fee=4.0, friction=1.0)  # c_review = 5
    #        fraud? amount  score
    y = np.array([1, 0, 1, 0])
    amt = np.array([100.0, 10.0, 50.0, 20.0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    # tau=0.5: alert where score>=0.5 -> rows 0,1.
    #   row0: alert & fraud  -> caught, 0
    #   row1: alert & legit  -> false alarm, +5
    #   row2: no-alert & fraud -> missed, +50
    #   row3: no-alert & legit -> 0
    assert expected_cost(y, amt, scores, 0.5, cp) == 55.0
    # tau=0.0: alert everything -> two false alarms (rows 1,3) = 10, no missed.
    assert expected_cost(y, amt, scores, 0.0, cp) == 10.0
    # tau=1.0: alert nothing -> miss both frauds = 150.
    assert expected_cost(y, amt, scores, 1.0, cp) == 150.0


def test_optimal_threshold_picks_minimum():
    cp = CostParams()
    rng = np.random.default_rng(0)
    n = 2000
    y = (rng.random(n) < 0.05).astype(int)
    amt = rng.uniform(1, 500, n)
    # Informative score: frauds score higher on average.
    scores = np.clip(0.1 + 0.6 * y + rng.normal(0, 0.2, n), 0, 1)
    tau = optimal_threshold(y, amt, scores, cp)
    taus, costs = sweep(y, amt, scores, cp)
    assert costs[np.argmin(np.abs(taus - tau))] == costs.min()
    # A sensible threshold sits strictly inside (0, 1), not at an extreme.
    assert 0.0 < tau < 1.0


if __name__ == "__main__":
    test_expected_cost_hand_example()
    print("PASS test_expected_cost_hand_example")
    test_optimal_threshold_picks_minimum()
    print("PASS test_optimal_threshold_picks_minimum")
