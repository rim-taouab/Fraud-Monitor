"""The three baseline rules. Every later model must beat these numbers.

ALL three are computed CAUSALLY: the decision for a transaction at time t uses
only transactions at time < t (or the current one's own fields). No rule peeks
at the future. This is the same leakage discipline the ML features must follow,
so we prove the pattern here first.

Rules implemented:
  1. velocity        - too many charges on one card in a short window
  2. impossible_travel - implied speed between consecutive card uses too high
  3. amount_spike    - charge far above the user's own past typical amount

Each returns a boolean Series aligned to df.index. `apply_rules` combines them
(OR) into a single alert, matching how a simple rule engine fires.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from generator import haversine_km
from schema import AMOUNT, CARD_ID, LAT, LON, MERCHANT_CATEGORY, TIMESTAMP, USER_ID


@dataclass
class RuleParams:
    velocity_window: str = "5min"
    velocity_min_prior: int = 4        # >=4 OTHER charges in the window => alert
    impossible_speed_kmh: float = 900.0  # faster than a commercial jet => alert
    impossible_min_km: float = 500.0   # ignore sub-500km hops (jitter, short trips)
    amount_spike_mult: float = 6.0     # amount > 6x past median => alert
    amount_min_history: int = 5        # need >=5 prior txns to judge "spike"


def velocity_rule(df: pd.DataFrame, p: RuleParams) -> pd.Series:
    """Count prior charges on the same card within the window (causal)."""
    out = pd.Series(False, index=df.index)
    # Rolling time window needs a sorted DatetimeIndex per card.
    for _, g in df.groupby(CARD_ID, sort=False):
        g = g.sort_values(TIMESTAMP)
        s = g.set_index(TIMESTAMP)[AMOUNT]
        # count() over the window includes the current row; subtract it to get
        # the number of PRIOR charges within [t-window, t].
        prior = s.rolling(p.velocity_window).count().to_numpy() - 1
        out.loc[g.index] = prior >= p.velocity_min_prior
    return out


def impossible_travel_rule(df: pd.DataFrame, p: RuleParams) -> pd.Series:
    """Implied km/h from the previous charge on the same card (causal)."""
    out = pd.Series(False, index=df.index)
    for _, g in df.groupby(CARD_ID, sort=False):
        g = g.sort_values(TIMESTAMP)
        dt_h = g[TIMESTAMP].diff().dt.total_seconds().to_numpy() / 3600.0
        dist = haversine_km(
            g[LAT].shift().to_numpy(), g[LON].shift().to_numpy(),
            g[LAT].to_numpy(), g[LON].to_numpy(),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = np.where(dt_h > 0, dist / dt_h, 0.0)
        # Require BOTH a real distance and an impossible speed: a 50km hop in a
        # minute is a GPS/merchant-location artifact, not teleportation.
        out.loc[g.index] = (speed > p.impossible_speed_kmh) & (dist > p.impossible_min_km)
    return out


def amount_spike_rule(df: pd.DataFrame, p: RuleParams) -> pd.Series:
    """Amount vs. the user's PAST median FOR THE SAME CATEGORY (causal).

    Comparing against a single cross-category median flags every legitimate
    travel/electronics purchase that follows a run of groceries. Judging a
    spike within its own merchant category removes that false-positive source
    while still catching a charge that is huge even for its category.
    """
    out = pd.Series(False, index=df.index)
    for _, g in df.groupby([USER_ID, MERCHANT_CATEGORY], sort=False):
        g = g.sort_values(TIMESTAMP)
        amt = g[AMOUNT]
        # expanding().median() includes the current row; shift(1) drops it so we
        # compare against same-category history only.
        past_median = amt.expanding().median().shift(1)
        past_count = amt.expanding().count().shift(1)
        flag = (past_count >= p.amount_min_history) & (amt > p.amount_spike_mult * past_median)
        out.loc[g.index] = flag.fillna(False).to_numpy()
    return out


def apply_rules(df: pd.DataFrame, p: RuleParams | None = None) -> pd.DataFrame:
    """Return df with per-rule boolean columns and a combined `rule_alert`."""
    p = p or RuleParams()
    res = pd.DataFrame(index=df.index)
    res["rule_velocity"] = velocity_rule(df, p)
    res["rule_impossible_travel"] = impossible_travel_rule(df, p)
    res["rule_amount_spike"] = amount_spike_rule(df, p)
    res["rule_alert"] = res.any(axis=1)
    return res
