"""Phase 4 entry point: the adversarial arms race.

    python src/run_phase4.py

An evolutionary fraudster probes the detector and learns to slip under its
threshold. We run two arenas in parallel against the SAME learning attacker
design:

  - FROZEN defense : the round-0 model, never updated. The attacker's skill
                     accumulates each round, so detection decays and profit climbs.
  - ADAPTIVE defense: retrained every round on the fraud it has just seen. It
                     keeps pace, so detection holds and profit stays low.

Phase 4 is "done when" one chart shows the frozen model decaying while the
retrained model holds, with the attacker's profit as the second line. This
computes the series behind that chart (make_chart_phase4.py draws it).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from attacker import EvolutionaryAttacker, build_victim_pool
from costs import CostParams, optimal_threshold
from features import ALL_FEATURES, MerchantRisk, add_causal_features
from generator import GenConfig, generate
from models import XGBDetector
from run_phase3 import three_way_split
from schema import (AMOUNT, CARD_ID, IS_FRAUD, LAT, LON, TIMESTAMP, USER_ID)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 100)


def run_arms_race(n_rounds=15, n_victims=40, seed=0, verbose=True):
    cp = CostParams(review_fee=4.0, friction=1.0)
    df = add_causal_features(generate(GenConfig(seed=7)))
    train, val, _ = three_way_split(df)

    mr = MerchantRisk().fit(train)
    for f in (train, val):
        f["merchant_risk"] = mr.transform(f)
    mr_global = mr.global_rate_

    Xtr = train[ALL_FEATURES].to_numpy()
    ytr = train[IS_FRAUD].to_numpy()
    a_tr = train[AMOUNT].to_numpy()
    w_tr = np.where(ytr == 1, a_tr, cp.c_review)

    # Round-0 cost-sensitive detector; tau* from validation (fixed thereafter so
    # the arms race is about the MODEL, not the threshold).
    det0 = XGBDetector().fit(Xtr, ytr, sample_weight=w_tr)
    tau = optimal_threshold(val[IS_FRAUD].to_numpy(), val[AMOUNT].to_numpy(),
                            det0.score(val[ALL_FEATURES].to_numpy()), cp)

    det_frozen = det0                       # never retrained
    det_adaptive = XGBDetector().fit(Xtr, ytr, sample_weight=w_tr)  # will retrain

    # Victim pools (one per arena), drawn from the training users' real history.
    normal = train[train[IS_FRAUD] == 0]
    start_ts = df[TIMESTAMP].min()
    rng = np.random.default_rng(seed)
    pool_f = build_victim_pool(normal, USER_ID, CARD_ID, TIMESTAMP, AMOUNT,
                               LAT, LON, start_ts, n_victims, rng)
    pool_a = build_victim_pool(normal, USER_ID, CARD_ID, TIMESTAMP, AMOUNT,
                               LAT, LON, start_ts, n_victims, rng)

    # One generation per round: the attacker's skill accrues gradually across
    # rounds (round 1 is barely optimized, so most attacks are caught), which is
    # what makes the frozen model's detection DECAY over time rather than snap to
    # its floor in a single round.
    atk_f = EvolutionaryAttacker(pop_size=20, generations=1, elite=5, seed=seed + 1)
    atk_a = EvolutionaryAttacker(pop_size=20, generations=1, elite=5, seed=seed + 2)

    # Accumulated adaptive-arena attacks (fed back into retraining).
    X_hist = [Xtr]; y_hist = [ytr]; w_hist = [w_tr]

    out = {"round": [], "det_frozen": [], "det_adaptive": [],
           "profit_frozen": [], "profit_adaptive": []}

    def value_detection(det, X, amt):
        """Value-weighted detection rate: caught value / total value."""
        caught = det.score(X) >= tau
        tot = amt.sum()
        return float(amt[caught].sum() / tot) if tot > 0 else 1.0

    for r in range(1, n_rounds + 1):
        # --- frozen arena: attacker keeps learning vs the static round-0 model
        _, Xf, af, prof_f = atk_f.step(det_frozen.score, tau, pool_f, mr_global)
        det_f = value_detection(det_frozen, Xf, af)

        # --- adaptive arena: attacker learns vs the CURRENT retrained model
        _, Xa, aa, prof_a = atk_a.step(det_adaptive.score, tau, pool_a, mr_global)
        det_a = value_detection(det_adaptive, Xa, aa)  # before retraining

        # defense adapts: retrain on everything seen, including this round's attack
        X_hist.append(Xa); y_hist.append(np.ones(len(Xa), int)); w_hist.append(aa)
        det_adaptive = XGBDetector().fit(
            np.vstack(X_hist), np.concatenate(y_hist),
            sample_weight=np.concatenate(w_hist))

        out["round"].append(r)
        out["det_frozen"].append(det_f)
        out["det_adaptive"].append(det_a)
        out["profit_frozen"].append(prof_f)
        out["profit_adaptive"].append(prof_a)
        if verbose:
            print(f"round {r:2d} | frozen: detect {det_f:5.1%} profit {prof_f:9.0f} "
                  f"| adaptive: detect {det_a:5.1%} profit {prof_a:9.0f}")

    out["tau"] = tau
    out["n_probes"] = atk_f.n_probes + atk_a.n_probes
    return out


def main() -> None:
    res = run_arms_race()
    df = pd.DataFrame({k: res[k] for k in
                       ["round", "det_frozen", "det_adaptive",
                        "profit_frozen", "profit_adaptive"]})
    print("\n" + "=" * 72)
    print(f"threshold tau (fixed) = {res['tau']:.3f} | attacker probes used = "
          f"{res['n_probes']:,}")
    print("=" * 72)
    f0, fN = df["det_frozen"].iloc[0], df["det_frozen"].iloc[-1]
    aN = df["det_adaptive"].iloc[-1]
    pf = df["profit_frozen"].sum()
    pa = df["profit_adaptive"].sum()
    print(f"frozen defense  : detection {f0:.0%} -> {fN:.0%} "
          f"(decayed {f0 - fN:+.0%}); attacker took {pf:,.0f} total")
    print(f"adaptive defense: detection holds ~{df['det_adaptive'].mean():.0%} "
          f"(final {aN:.0%}); attacker took {pa:,.0f} total")
    if pa > 0:
        print(f"\n=> retraining cut the attacker's haul by "
              f"{(1 - pa / max(pf, 1)):.0%} versus leaving the model frozen.")


if __name__ == "__main__":
    main()
