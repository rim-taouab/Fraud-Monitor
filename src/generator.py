"""Synthetic transaction generator.

Produces a time-ordered DataFrame matching src/schema.COLUMNS: realistic
per-user normal behavior with four injected fraud patterns. Everything is
seeded and reproducible.

The fraud injectors are deliberately separated from normal generation and
each tagged with its FraudPattern, because:
  - Phase 1 rules target these exact patterns (and we measure recall per pattern)
  - Phase 4's attacker mutates these knobs to slip under the detector

Geography note: we use a flat-earth haversine so "impossible travel" is a real,
checkable physical constraint (km / hours => implied km/h).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from schema import (
    AMOUNT, CARD_ID, COLUMNS, DEVICE_ID, FRAUD_PATTERN, IS_FRAUD, LAT, LON,
    MERCHANT_CATEGORY, MERCHANT_ID, TIMESTAMP, TXN_ID, USER_ID,
    FraudPattern, MerchantCategory,
)

EARTH_RADIUS_KM = 6371.0

# Typical ticket size (lognormal mean in currency units) per category. Drives
# how "amount spike" and category choice look. TRAVEL/ELECTRONICS run high.
_CATEGORY_TICKET = {
    MerchantCategory.GROCERY: 45.0,
    MerchantCategory.RESTAURANT: 35.0,
    MerchantCategory.FUEL: 55.0,
    MerchantCategory.RETAIL: 60.0,
    MerchantCategory.TRAVEL: 320.0,
    MerchantCategory.ELECTRONICS: 240.0,
    MerchantCategory.ATM: 120.0,
    MerchantCategory.ONLINE: 50.0,
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Vectorized over numpy arrays or scalars."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@dataclass
class UserProfile:
    user_id: int
    home_lat: float
    home_lon: float
    card_ids: list[int]
    device_ids: list[int]
    spend_scale: float                 # per-user multiplier on category tickets
    daily_txn_rate: float              # expected normal txns per day
    category_weights: np.ndarray       # preference over MerchantCategory


@dataclass
class GenConfig:
    n_users: int = 3000
    n_days: int = 90
    n_merchants: int = 500
    # Share of users attacked. Set so the fraud rate stays realistic (~0.3%,
    # near Kaggle's 0.17%) while giving ENOUGH of each pattern -- especially the
    # rare high-value ones -- for a stable, learnable signal and a cost estimate
    # that isn't driven by 2-3 lucky frauds.
    fraud_user_fraction: float = 0.09
    seed: int = 7
    start: str = "2026-01-01"
    # how many fraud episodes of each pattern, as a fraction of attacked users.
    # amount_spike / impossible_travel are weighted up: they are the high-value,
    # low-count patterns that dominate cost, so they need volume.
    pattern_mix: dict = field(default_factory=lambda: {
        FraudPattern.VELOCITY_BURST: 0.25,
        FraudPattern.IMPOSSIBLE_TRAVEL: 0.25,
        FraudPattern.AMOUNT_SPIKE: 0.30,
        FraudPattern.CARD_TESTING: 0.20,
    })


# --------------------------------------------------------------------------- #
# Normal behavior
# --------------------------------------------------------------------------- #
def _make_users(cfg: GenConfig, rng: np.random.Generator) -> list[UserProfile]:
    users = []
    for uid in range(cfg.n_users):
        # Homes scattered across a continent-sized box (degrees).
        home_lat = rng.uniform(25.0, 49.0)
        home_lon = rng.uniform(-123.0, -71.0)
        n_cards = rng.integers(1, 3, endpoint=True)
        n_devices = rng.integers(1, 2, endpoint=True)
        cat_w = rng.dirichlet(np.ones(len(MerchantCategory)) * 2.0)
        users.append(UserProfile(
            user_id=uid,
            home_lat=home_lat,
            home_lon=home_lon,
            card_ids=[uid * 10 + k for k in range(n_cards)],
            device_ids=[uid * 10 + k for k in range(n_devices)],
            spend_scale=float(rng.lognormal(0.0, 0.4)),
            daily_txn_rate=float(rng.uniform(0.3, 4.0)),
            category_weights=cat_w,
        ))
    return users


def _normal_txns(cfg, users, merch_lat, merch_lon, rng) -> pd.DataFrame:
    start = pd.Timestamp(cfg.start)
    horizon_s = cfg.n_days * 24 * 3600
    rows = []
    for u in users:
        n = rng.poisson(u.daily_txn_rate * cfg.n_days)
        if n == 0:
            continue
        # Transaction times uniform over the horizon (day-of-week/hour effects
        # omitted for Phase 1; add later if needed).
        secs = rng.integers(0, horizon_s, size=n)
        cats = rng.choice(len(MerchantCategory), size=n, p=u.category_weights)
        for s, cat in zip(secs, cats):
            cat = MerchantCategory(int(cat))
            ticket = _CATEGORY_TICKET[cat] * u.spend_scale
            amount = float(rng.lognormal(np.log(max(ticket, 1.0)), 0.45))
            merch = int(rng.integers(0, cfg.n_merchants))
            # Online/travel can be far from home; in-person sits near home.
            if cat in (MerchantCategory.ONLINE,):
                lat, lon = u.home_lat, u.home_lon  # CNP: billing ~ home
            elif cat == MerchantCategory.TRAVEL:
                # A real trip, but a coherent one (~100km radius), not a
                # continent-hop between consecutive charges.
                lat = u.home_lat + rng.normal(0, 1.0)
                lon = u.home_lon + rng.normal(0, 1.0)
            else:
                lat = u.home_lat + rng.normal(0, 0.15)
                lon = u.home_lon + rng.normal(0, 0.15)
            rows.append((
                start + pd.Timedelta(seconds=int(s)),
                u.user_id, int(rng.choice(u.card_ids)), merch, int(cat),
                round(amount, 2), float(lat), float(lon),
                int(rng.choice(u.device_ids)), 0, FraudPattern.NONE,
            ))
    cols = [TIMESTAMP, USER_ID, CARD_ID, MERCHANT_ID, MERCHANT_CATEGORY,
            AMOUNT, LAT, LON, DEVICE_ID, IS_FRAUD, FRAUD_PATTERN]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Fraud injectors — each returns a list of row-tuples in the same column order
# --------------------------------------------------------------------------- #
def _fraud_device(rng) -> int:
    # Fraud rides on a small pool of shared "mule" devices => Phase 4 graph rings.
    return int(rng.integers(900_000, 900_050))


def _inject_velocity_burst(u, cfg, rng):
    start = pd.Timestamp(cfg.start)
    t0 = start + pd.Timedelta(seconds=int(rng.integers(0, cfg.n_days * 86400)))
    card = int(rng.choice(u.card_ids))
    dev = _fraud_device(rng)
    n = int(rng.integers(6, 15))
    rows = []
    for _ in range(n):
        t0 += pd.Timedelta(seconds=int(rng.integers(5, 90)))  # seconds apart
        cat = MerchantCategory.ELECTRONICS
        amt = float(rng.lognormal(np.log(_CATEGORY_TICKET[cat]), 0.5))
        rows.append((t0, u.user_id, card, int(rng.integers(0, cfg.n_merchants)),
                     int(cat), round(amt, 2), u.home_lat, u.home_lon, dev,
                     1, FraudPattern.VELOCITY_BURST))
    return rows


def _inject_impossible_travel(u, cfg, rng):
    start = pd.Timestamp(cfg.start)
    t0 = start + pd.Timedelta(seconds=int(rng.integers(0, cfg.n_days * 86400)))
    card = int(rng.choice(u.card_ids))
    dev = _fraud_device(rng)
    # A home charge, then minutes later a charge thousands of km away.
    far_lat = u.home_lat + rng.choice([-1, 1]) * rng.uniform(20, 35)
    far_lon = u.home_lon + rng.choice([-1, 1]) * rng.uniform(40, 70)
    cat = MerchantCategory.RETAIL
    amt1 = float(rng.lognormal(np.log(_CATEGORY_TICKET[cat]), 0.4))
    amt2 = float(rng.lognormal(np.log(_CATEGORY_TICKET[cat]), 0.4))
    t1 = t0 + pd.Timedelta(minutes=int(rng.integers(10, 50)))
    return [
        (t0, u.user_id, card, int(rng.integers(0, cfg.n_merchants)), int(cat),
         round(amt1, 2), u.home_lat, u.home_lon, dev, 1,
         FraudPattern.IMPOSSIBLE_TRAVEL),
        (t1, u.user_id, card, int(rng.integers(0, cfg.n_merchants)), int(cat),
         round(amt2, 2), float(far_lat), float(far_lon), dev, 1,
         FraudPattern.IMPOSSIBLE_TRAVEL),
    ]


def _inject_amount_spike(u, cfg, rng):
    start = pd.Timestamp(cfg.start)
    t0 = start + pd.Timedelta(seconds=int(rng.integers(0, cfg.n_days * 86400)))
    card = int(rng.choice(u.card_ids))
    dev = _fraud_device(rng)
    cat = MerchantCategory.ELECTRONICS
    base = _CATEGORY_TICKET[cat] * u.spend_scale
    # 2-3 large charges spaced hours apart (not a velocity burst): the signal is
    # the AMOUNT vs the user's norm, not the timing. Multi-charge so the pattern
    # is learnable and shows up in enough test rows to measure its cost.
    n = int(rng.integers(2, 4))
    rows = []
    for _ in range(n):
        t0 += pd.Timedelta(hours=int(rng.integers(1, 9)))
        amt = float(base * rng.uniform(8, 20))  # far above the user's norm
        rows.append((t0, u.user_id, card, int(rng.integers(0, cfg.n_merchants)),
                     int(cat), round(amt, 2), u.home_lat, u.home_lon, dev, 1,
                     FraudPattern.AMOUNT_SPIKE))
    return rows


def _inject_card_testing(u, cfg, rng):
    start = pd.Timestamp(cfg.start)
    t0 = start + pd.Timedelta(seconds=int(rng.integers(0, cfg.n_days * 86400)))
    card = int(rng.choice(u.card_ids))
    dev = _fraud_device(rng)
    n = int(rng.integers(8, 20))
    rows = []
    for _ in range(n):
        t0 += pd.Timedelta(seconds=int(rng.integers(20, 180)))
        amt = round(float(rng.uniform(0.5, 3.0)), 2)  # tiny probe charges
        rows.append((t0, u.user_id, card, int(rng.integers(0, cfg.n_merchants)),
                     int(MerchantCategory.ONLINE), amt, u.home_lat, u.home_lon,
                     dev, 1, FraudPattern.CARD_TESTING))
    return rows


_INJECTORS = {
    FraudPattern.VELOCITY_BURST: _inject_velocity_burst,
    FraudPattern.IMPOSSIBLE_TRAVEL: _inject_impossible_travel,
    FraudPattern.AMOUNT_SPIKE: _inject_amount_spike,
    FraudPattern.CARD_TESTING: _inject_card_testing,
}


def _fraud_txns(cfg, users, rng) -> pd.DataFrame:
    n_attacked = max(1, int(cfg.n_users * cfg.fraud_user_fraction))
    victims = rng.choice(users, size=n_attacked, replace=False)
    patterns = list(cfg.pattern_mix.keys())
    probs = np.array([cfg.pattern_mix[p] for p in patterns])
    probs = probs / probs.sum()
    rows = []
    for u in victims:
        pat = patterns[int(rng.choice(len(patterns), p=probs))]
        rows.extend(_INJECTORS[pat](u, cfg, rng))
    cols = [TIMESTAMP, USER_ID, CARD_ID, MERCHANT_ID, MERCHANT_CATEGORY,
            AMOUNT, LAT, LON, DEVICE_ID, IS_FRAUD, FRAUD_PATTERN]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate(cfg: GenConfig | None = None) -> pd.DataFrame:
    """Return a reproducible, time-ordered transaction DataFrame."""
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(cfg.seed)
    users = _make_users(cfg, rng)
    merch_lat = rng.uniform(25.0, 49.0, size=cfg.n_merchants)
    merch_lon = rng.uniform(-123.0, -71.0, size=cfg.n_merchants)

    normal = _normal_txns(cfg, users, merch_lat, merch_lon, rng)
    fraud = _fraud_txns(cfg, users, rng)

    df = pd.concat([normal, fraud], ignore_index=True)
    df = df.sort_values(TIMESTAMP, kind="mergesort").reset_index(drop=True)
    df.insert(0, TXN_ID, np.arange(len(df), dtype=np.int64))
    # Enforce the canonical column order / dtypes from the schema.
    df = df[COLUMNS]
    df[IS_FRAUD] = df[IS_FRAUD].astype("int8")
    return df


if __name__ == "__main__":
    d = generate()
    rate = d[IS_FRAUD].mean()
    print(f"rows={len(d):,}  fraud={d[IS_FRAUD].sum():,}  rate={rate:.4%}")
    print(d[FRAUD_PATTERN].value_counts())
