"""Causal feature engineering for the ML detector (Phase 2).

Two kinds of features, split by their leakage risk:

1. SEQUENCE features (add_causal_features): rolling counts/amounts per card,
   time-since-last, distance & implied speed, ratio to the user's past median,
   time-of-day. These use only transactions at time < t (past-only), so they are
   safe to compute on the FULL frame before the train/test split -- a feature at
   time t is information genuinely available at inference time.

   The leakage discipline: every rolling/expanding window is `closed="left"`
   (excludes the current row) or explicitly `.shift(1)`-ed. The same
   test_rules_are_causal idea extends here (see test_features.py).

2. LABEL-DERIVED feature (MerchantRisk): the historical fraud rate per merchant.
   This DOES use is_fraud, so it must be fit on TRAIN ONLY and then applied to
   test. Fitting it on the full data would leak test labels into the features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from generator import haversine_km
from schema import (
    AMOUNT, CARD_ID, IS_FRAUD, LAT, LON, MERCHANT_CATEGORY, MERCHANT_ID,
    TIMESTAMP, USER_ID,
)

# The columns the models consume. Keeps feature construction and model input in
# sync and makes it obvious that no label column sneaks in.
SEQUENCE_FEATURES = [
    "log_amount",
    "hour",
    "secs_since_prev_card",
    "dist_prev_card_km",
    "speed_kmh",
    "cnt_card_1min",
    "cnt_card_1h",
    "cnt_card_24h",
    "amt_card_1h",
    "amt_card_24h",
    "amount_ratio_user_median",
    MERCHANT_CATEGORY,  # treated as a numeric code; low-cardinality, fine for trees
]
MERCHANT_RISK_FEATURE = "merchant_risk"
ALL_FEATURES = SEQUENCE_FEATURES + [MERCHANT_RISK_FEATURE]


def _rolling_count_left(times: pd.Series, window: str) -> np.ndarray:
    """Number of PRIOR rows within `window` (current row excluded)."""
    s = pd.Series(1, index=times)
    return s.rolling(window, closed="left").sum().to_numpy()


def _rolling_sum_left(times: pd.Series, values: pd.Series, window: str) -> np.ndarray:
    """Sum of a value over PRIOR rows within `window` (current row excluded)."""
    s = pd.Series(values.to_numpy(), index=times)
    return s.rolling(window, closed="left").sum().to_numpy()


def add_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the causal sequence features added.

    Safe on the full dataset: every feature uses only past transactions.
    """
    df = df.sort_values(TIMESTAMP, kind="mergesort").reset_index(drop=True)
    out = df.copy()

    out["log_amount"] = np.log1p(df[AMOUNT].to_numpy())
    out["hour"] = df[TIMESTAMP].dt.hour.to_numpy()

    # --- per-card sequence features ---
    secs_prev = np.full(len(df), np.nan)
    dist_prev = np.full(len(df), np.nan)
    speed = np.zeros(len(df))
    cnt_1min = np.zeros(len(df))
    cnt_1h = np.zeros(len(df))
    cnt_24h = np.zeros(len(df))
    amt_1h = np.zeros(len(df))
    amt_24h = np.zeros(len(df))

    for _, g in df.groupby(CARD_ID, sort=False):
        g = g.sort_values(TIMESTAMP)
        idx = g.index.to_numpy()
        t = g[TIMESTAMP]

        dt_s = t.diff().dt.total_seconds().to_numpy()
        secs_prev[idx] = dt_s

        d = haversine_km(g[LAT].shift().to_numpy(), g[LON].shift().to_numpy(),
                         g[LAT].to_numpy(), g[LON].to_numpy())
        dist_prev[idx] = d
        with np.errstate(divide="ignore", invalid="ignore"):
            sp = np.where(dt_s > 0, d / (dt_s / 3600.0), 0.0)
        speed[idx] = np.nan_to_num(sp, nan=0.0, posinf=0.0)

        cnt_1min[idx] = _rolling_count_left(t, "1min")
        cnt_1h[idx] = _rolling_count_left(t, "1h")
        cnt_24h[idx] = _rolling_count_left(t, "24h")
        amt_1h[idx] = _rolling_sum_left(t, g[AMOUNT], "1h")
        amt_24h[idx] = _rolling_sum_left(t, g[AMOUNT], "24h")

    # First-seen rows have no prior txn: large gap, zero distance/speed.
    out["secs_since_prev_card"] = np.nan_to_num(secs_prev, nan=1e7)
    out["dist_prev_card_km"] = np.nan_to_num(dist_prev, nan=0.0)
    out["speed_kmh"] = speed
    out["cnt_card_1min"] = np.nan_to_num(cnt_1min, nan=0.0)
    out["cnt_card_1h"] = np.nan_to_num(cnt_1h, nan=0.0)
    out["cnt_card_24h"] = np.nan_to_num(cnt_24h, nan=0.0)
    out["amt_card_1h"] = np.nan_to_num(amt_1h, nan=0.0)
    out["amt_card_24h"] = np.nan_to_num(amt_24h, nan=0.0)

    # --- per-user ratio to past median amount (causal) ---
    ratio = np.ones(len(df))
    for _, g in df.groupby(USER_ID, sort=False):
        g = g.sort_values(TIMESTAMP)
        amt = g[AMOUNT]
        past_median = amt.expanding().median().shift(1)
        r = (amt / past_median).to_numpy()
        ratio[g.index.to_numpy()] = np.nan_to_num(r, nan=1.0, posinf=1.0)
    out["amount_ratio_user_median"] = ratio

    return out


class MerchantRisk:
    """Historical fraud rate per merchant, smoothed toward the global rate.

    FIT ON TRAIN ONLY. Uses is_fraud, so fitting on test would leak labels.
    Unseen merchants fall back to the global (train) fraud rate.
    """

    def __init__(self, smoothing: float = 20.0):
        self.smoothing = smoothing
        self.global_rate_ = 0.0
        self.rate_: dict = {}

    def fit(self, train: pd.DataFrame) -> "MerchantRisk":
        self.global_rate_ = float(train[IS_FRAUD].mean())
        grp = train.groupby(MERCHANT_ID)[IS_FRAUD]
        n = grp.count()
        k = grp.sum()
        # Bayesian shrink: (k + a) / (n + a/p) style smoothing toward global rate.
        smoothed = (k + self.smoothing * self.global_rate_) / (n + self.smoothing)
        self.rate_ = smoothed.to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return df[MERCHANT_ID].map(self.rate_).fillna(self.global_rate_).to_numpy()
