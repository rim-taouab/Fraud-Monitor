"""The decision layer: turn fraud scores into alert/allow decisions by the
threshold that MINIMIZES expected cost, not the one that maximizes F1.

Cost model (from the build plan):

    C(tau) = sum_i [ 1(p_i <  tau) * y_i     * a_i        # missed fraud: lose the amount
                   + 1(p_i >= tau) * (1-y_i) * c_review ] # false alarm: pay a review fee

Lowering tau raises alerts -> fewer missed frauds (less amount lost) but more
false alarms (more review fees). The minimum balances the two. A caught fraud
and a correctly-allowed transaction cost 0 in this model, matching the plan.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostParams:
    review_fee: float = 4.0    # analyst time to review one alert
    friction: float = 1.0      # customer friction of a false alarm
    # A missed fraud costs the transaction amount (a_i); no fixed term.

    @property
    def c_review(self) -> float:
        return self.review_fee + self.friction


def expected_cost(y, amounts, scores, tau: float, cp: CostParams) -> float:
    """Total C(tau) over the given transactions."""
    y = np.asarray(y)
    amounts = np.asarray(amounts, dtype=float)
    alert = np.asarray(scores) >= tau
    missed_fraud = (~alert) & (y == 1)
    false_alarm = alert & (y == 0)
    return float(amounts[missed_fraud].sum() + cp.c_review * false_alarm.sum())


def cost_per_1000(y, amounts, scores, tau: float, cp: CostParams) -> float:
    return expected_cost(y, amounts, scores, tau, cp) / len(y) * 1000.0


def binary_cost(y, amounts, alert, cp: CostParams) -> float:
    """Cost of a fixed binary policy (e.g. the rule engine's alert column)."""
    y = np.asarray(y)
    amounts = np.asarray(amounts, dtype=float)
    alert = np.asarray(alert).astype(bool)
    missed_fraud = (~alert) & (y == 1)
    false_alarm = alert & (y == 0)
    return float(amounts[missed_fraud].sum() + cp.c_review * false_alarm.sum())


def sweep(y, amounts, scores, cp: CostParams, n_grid: int = 300):
    """Return (taus, costs_per_1000) sweeping tau across the score range."""
    taus = np.linspace(0.0, 1.0, n_grid)
    costs = np.array([cost_per_1000(y, amounts, scores, t, cp) for t in taus])
    return taus, costs


def optimal_threshold(y, amounts, scores, cp: CostParams, n_grid: int = 300) -> float:
    """tau* minimizing expected cost on the GIVEN data (use validation, not test)."""
    taus, costs = sweep(y, amounts, scores, cp, n_grid)
    return float(taus[int(np.argmin(costs))])


def bootstrap_savings(y, amounts, scores, rule_alert, tau: float, cp: CostParams,
                      n_boot: int = 2000, seed: int = 0):
    """Bootstrap the model's savings vs the rules, in cost-per-1000 units.

    The policy is FIXED (tau chosen on validation, rules fixed); we only resample
    the test transactions to get a sampling distribution of the savings. Returns
    (mean, lo95, hi95, point_model, point_rules).
    """
    y = np.asarray(y)
    amounts = np.asarray(amounts, dtype=float)
    scores = np.asarray(scores)
    rule_alert = np.asarray(rule_alert).astype(bool)
    n = len(y)
    rng = np.random.default_rng(seed)

    def savings_on(idx):
        m = cost_per_1000(y[idx], amounts[idx], scores[idx], tau, cp)
        r = binary_cost(y[idx], amounts[idx], rule_alert[idx], cp) / len(idx) * 1000.0
        return r - m  # positive => model is cheaper

    point_model = cost_per_1000(y, amounts, scores, tau, cp)
    point_rules = binary_cost(y, amounts, rule_alert, cp) / n * 1000.0

    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = savings_on(idx)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(draws.mean()), float(lo), float(hi), point_model, point_rules


def per_band_thresholds(y_val, amt_val, score_val, cp: CostParams, bands):
    """Pick a separate tau per amount band on validation data.

    `bands` is a list of (lo, hi) amount edges. Returns {(_lo,_hi): tau}.
    """
    out = {}
    amt_val = np.asarray(amt_val, dtype=float)
    for lo, hi in bands:
        m = (amt_val >= lo) & (amt_val < hi)
        if m.sum() < 50 or np.asarray(y_val)[m].sum() == 0:
            out[(lo, hi)] = optimal_threshold(y_val, amt_val, score_val, cp)
        else:
            out[(lo, hi)] = optimal_threshold(
                np.asarray(y_val)[m], amt_val[m], np.asarray(score_val)[m], cp)
    return out


def apply_per_band(scores, amounts, band_taus) -> np.ndarray:
    """Alert decision using the band-specific threshold for each transaction."""
    scores = np.asarray(scores)
    amounts = np.asarray(amounts, dtype=float)
    alert = np.zeros(len(scores), dtype=bool)
    for (lo, hi), tau in band_taus.items():
        m = (amounts >= lo) & (amounts < hi)
        alert[m] = scores[m] >= tau
    return alert
