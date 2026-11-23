"""Evolutionary fraudster for the Phase 4 arms race.

The attacker searches the space of fraud *strategies* (how big a charge, how
many, how far apart, how far from home, which merchant category) to maximize
UNDETECTED value against a given detector. It only ever queries the detector's
score (a black box) -- a realistic probing attacker, not one with gradients.

The hard part is scoring a candidate fraud transaction the way the live system
would: through the SAME causal features the detector was trained on. We build
those features analytically from each victim's real history (counts in rolling
windows via searchsorted, time-since-last, distance/speed, ratio to the user's
median), so the inner evolutionary loop stays fast.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from features import ALL_FEATURES
from generator import haversine_km
from schema import MERCHANT_CATEGORY

# Genome layout (continuous where possible; a few are rounded at use time).
#   amount_mult  : charge size as a multiple of the victim's median spend
#   burst_size   : number of charges in the episode
#   gap_s        : seconds between charges
#   distance_km  : how far from home the charge is placed
#   category     : merchant category code
GENOME_KEYS = ["amount_mult", "burst_size", "gap_s", "distance_km", "category"]
BOUNDS = {
    "amount_mult": (1.5, 40.0),
    "burst_size": (1.0, 6.0),
    "gap_s": (10.0, 6 * 3600.0),
    "distance_km": (0.0, 3000.0),
    "category": (0.0, 7.0),
}
_WINDOWS = {"1min": 60.0, "1h": 3600.0, "24h": 86400.0}
_HALF_DAY = 12 * 3600.0


@dataclass
class VictimPool:
    """Precomputed per-victim context, enough to featurize any attack cheaply."""
    ts: list            # sorted normal txn times (seconds since start) per victim
    amt_prefix: list    # prefix sums of amounts aligned to ts (len n+1) per victim
    last_t: np.ndarray
    home_lat: np.ndarray
    home_lon: np.ndarray
    last_lat: np.ndarray
    last_lon: np.ndarray
    median_amt: np.ndarray

    def __len__(self):
        return len(self.ts)


def build_victim_pool(df_normal, user_col, card_col, time_col, amt_col,
                      lat_col, lon_col, start_ts, n_victims, rng) -> VictimPool:
    """Sample victim cards and precompute their context arrays."""
    cards = df_normal[card_col].unique()
    chosen = rng.choice(cards, size=min(n_victims, len(cards)), replace=False)
    ts, amt_prefix = [], []
    last_t, hlat, hlon, llat, llon, med = [], [], [], [], [], []
    for c in chosen:
        g = df_normal[df_normal[card_col] == c].sort_values(time_col)
        if len(g) < 5:
            continue
        t = (g[time_col] - start_ts).dt.total_seconds().to_numpy()
        a = g[amt_col].to_numpy()
        ts.append(t)
        amt_prefix.append(np.concatenate([[0.0], np.cumsum(a)]))
        last_t.append(t[-1])
        hlat.append(g[lat_col].iloc[0]); hlon.append(g[lon_col].iloc[0])
        llat.append(g[lat_col].iloc[-1]); llon.append(g[lon_col].iloc[-1])
        med.append(np.median(a))
    return VictimPool(ts, amt_prefix, np.array(last_t), np.array(hlat),
                      np.array(hlon), np.array(llat), np.array(llon), np.array(med))


def _count_sum_before(t_arr, prefix, t_hi, w):
    """(#normal txns, sum amount) in [t_hi - w, t_hi) via searchsorted."""
    lo = np.searchsorted(t_arr, t_hi - w, side="left")
    hi = np.searchsorted(t_arr, t_hi, side="left")
    return hi - lo, prefix[hi] - prefix[lo]


def attack_rows(genome: dict, pool: VictimPool, mr_global: float):
    """Build the (features, amounts) for applying `genome` to every victim.

    Returns (X [n_rows, n_features], amounts [n_rows]) with X columns in
    ALL_FEATURES order, ready to score with any trained detector.
    """
    k = int(round(np.clip(genome["burst_size"], *BOUNDS["burst_size"])))
    gap = float(np.clip(genome["gap_s"], *BOUNDS["gap_s"]))
    mult = float(np.clip(genome["amount_mult"], *BOUNDS["amount_mult"]))
    dist = float(np.clip(genome["distance_km"], *BOUNDS["distance_km"]))
    cat = int(round(np.clip(genome["category"], *BOUNDS["category"])))
    start_gap = 3600.0  # first charge 1h after the victim's last normal txn

    cols = {c: [] for c in ALL_FEATURES}
    amounts = []
    # Convert a km offset to a rough degree offset in latitude.
    dlat = dist / 111.0

    for i in range(len(pool)):
        t_last = pool.last_t[i]
        amount = mult * pool.median_amt[i]
        fraud_lat = pool.home_lat[i] + dlat
        fraud_lon = pool.home_lon[i]
        burst_times = t_last + start_gap + gap * np.arange(k)
        for j in range(k):
            tj = burst_times[j]
            # time since previous charge on this card (normal or earlier burst)
            prev_t = t_last if j == 0 else burst_times[j - 1]
            secs_prev = tj - prev_t
            # distance/speed from previous location
            if j == 0:
                d = haversine_km(pool.last_lat[i], pool.last_lon[i], fraud_lat, fraud_lon)
            else:
                d = 0.0  # same place as the previous burst charge
            speed = d / (secs_prev / 3600.0) if secs_prev > 0 else 0.0

            feat = {}
            feat["log_amount"] = np.log1p(amount)
            feat["hour"] = int((tj // 3600) % 24)
            feat["secs_since_prev_card"] = secs_prev
            feat["dist_prev_card_km"] = d
            feat["speed_kmh"] = speed
            for wname, w in _WINDOWS.items():
                cnt, ssum = _count_sum_before(pool.ts[i], pool.amt_prefix[i], tj, w)
                # earlier burst charges that fall inside this window
                extra = min(j, int(w // gap)) if gap > 0 else j
                if wname == "1min":
                    feat["cnt_card_1min"] = cnt + extra
                elif wname == "1h":
                    feat["cnt_card_1h"] = cnt + extra
                    feat["amt_card_1h"] = ssum + extra * amount
                else:
                    feat["cnt_card_24h"] = cnt + extra
                    feat["amt_card_24h"] = ssum + extra * amount
            feat["amount_ratio_user_median"] = amount / max(pool.median_amt[i], 1e-9)
            feat[MERCHANT_CATEGORY] = cat
            feat["merchant_risk"] = mr_global

            for c in ALL_FEATURES:
                cols[c].append(feat[c])
            amounts.append(amount)

    X = np.column_stack([np.asarray(cols[c], dtype=float) for c in ALL_FEATURES])
    return X, np.asarray(amounts)


class EvolutionaryAttacker:
    """Black-box evolutionary search for the most profitable evasive strategy."""

    def __init__(self, pop_size=24, generations=6, elite=6, seed=0):
        self.pop_size = pop_size
        self.generations = generations
        self.elite = elite
        self.rng = np.random.default_rng(seed)
        self.n_probes = 0  # detector queries used (budget accounting)
        self.pop = None    # persistent population: the attacker KEEPS learning
                           # across rounds, so its skill accumulates with budget.

    def _random_genome(self):
        return {k: float(self.rng.uniform(*BOUNDS[k])) for k in GENOME_KEYS}

    def _mutate(self, g):
        child = dict(g)
        for k in GENOME_KEYS:
            lo, hi = BOUNDS[k]
            child[k] = float(np.clip(g[k] + self.rng.normal(0, 0.15) * (hi - lo), lo, hi))
        return child

    def _crossover(self, a, b):
        return {k: (a[k] if self.rng.random() < 0.5 else b[k]) for k in GENOME_KEYS}

    def step(self, score_fn, tau, pool, mr_global):
        """Run `generations` of evolution against the CURRENT defense, carrying
        the population forward from previous rounds.

        Returns (best_genome, best_X, best_amounts, best_fitness) for this round.
        score_fn: X -> fraud scores in [0,1]. fitness = undetected value.
        """
        if self.pop is None:
            self.pop = [self._random_genome() for _ in range(self.pop_size)]
        best = None
        for _ in range(self.generations):
            scored = []
            for g in self.pop:
                X, amt = attack_rows(g, pool, mr_global)
                s = score_fn(X)
                self.n_probes += len(s)
                undetected_value = float(amt[s < tau].sum())
                scored.append((undetected_value, g, X, amt))
            scored.sort(key=lambda r: r[0], reverse=True)
            if best is None or scored[0][0] > best[0]:
                best = scored[0]
            elites = [r[1] for r in scored[:self.elite]]
            self.pop = list(elites)
            while len(self.pop) < self.pop_size:
                a, b = self.rng.choice(len(elites), size=2)
                self.pop.append(self._mutate(self._crossover(elites[a], elites[b])))
        return best[1], best[2], best[3], best[0]
