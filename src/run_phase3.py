"""Phase 3 entry point: the quant decision layer.

    python src/run_phase3.py

Picks the cost-minimizing alert threshold on a VALIDATION window (never on
test), then reports on the held-out test set:
  - cost per 1,000 transactions for the model vs the rules, with a 95% bootstrap CI
  - whether a per-amount-band threshold beats a single global threshold
  - SHAP top-3 reasons for a sample of alerts

Phase 3 is "done when" we can say: "this threshold saves X per 1,000 transactions
versus the rules, with a 95% interval." This prints exactly that.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from costs import (CostParams, apply_per_band, binary_cost, bootstrap_savings,
                   cost_per_1000, optimal_threshold, per_band_thresholds)
from explain import shap_contributions, top_reasons
from features import ALL_FEATURES, MerchantRisk, add_causal_features
from generator import GenConfig, generate
from metrics import prf
from models import XGBDetector
from rules import apply_rules
from schema import AMOUNT, IS_FRAUD, TIMESTAMP
from split import time_split

warnings.filterwarnings("ignore")
pd.set_option("display.width", 100)


def three_way_split(df, val_start=0.60, test_start=0.75):
    df = df.sort_values(TIMESTAMP, kind="mergesort").reset_index(drop=True)
    q_val = df[TIMESTAMP].quantile(val_start)
    q_test = df[TIMESTAMP].quantile(test_start)
    train = df[df[TIMESTAMP] < q_val].reset_index(drop=True)
    val = df[(df[TIMESTAMP] >= q_val) & (df[TIMESTAMP] < q_test)].reset_index(drop=True)
    test = df[df[TIMESTAMP] >= q_test].reset_index(drop=True)
    return train, val, test


def main() -> None:
    cp = CostParams(review_fee=4.0, friction=1.0)  # c_review = 5.0

    df = add_causal_features(generate(GenConfig(seed=7)))
    train, val, test = three_way_split(df)

    mr = MerchantRisk().fit(train)
    for frame in (train, val, test):
        frame["merchant_risk"] = mr.transform(frame)

    Xtr, ytr = train[ALL_FEATURES].to_numpy(), train[IS_FRAUD].to_numpy()
    a_tr = train[AMOUNT].to_numpy()

    # Two detectors on identical data:
    #  - naive: trained to classify (every fraud weighted equally)
    #  - cost-sensitive: each fraud weighted by its AMOUNT, each legit by
    #    c_review, so training approximates the business cost directly. This is
    #    the fix for "great PR-AUC but no cost savings": the threshold alone
    #    cannot repair a model optimized for the wrong objective.
    det = XGBDetector().fit(Xtr, ytr)
    w = np.where(ytr == 1, a_tr, cp.c_review)
    det_c = XGBDetector().fit(Xtr, ytr, sample_weight=w)

    Xva, Xte = val[ALL_FEATURES].to_numpy(), test[ALL_FEATURES].to_numpy()
    s_val, s_test = det.score(Xva), det.score(Xte)
    sc_val, sc_test = det_c.score(Xva), det_c.score(Xte)

    y_test = test[IS_FRAUD].to_numpy()
    a_test = test[AMOUNT].to_numpy()
    rule_alert_test = apply_rules(test)["rule_alert"].to_numpy()

    print("=" * 72)
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}")
    print(f"cost model: missed fraud = amount; false alarm = c_review = "
          f"{cp.c_review:.2f} (fee {cp.review_fee} + friction {cp.friction})")
    print("=" * 72)

    # Cost-minimizing threshold for EACH model, chosen on VALIDATION.
    y_val, a_val = val[IS_FRAUD].to_numpy(), val[AMOUNT].to_numpy()
    tau = optimal_threshold(y_val, a_val, s_val, cp)
    tau_c = optimal_threshold(y_val, a_val, sc_val, cp)

    c_allow = binary_cost(y_test, a_test, np.zeros(len(test)), cp) / len(test) * 1000
    c_rules = binary_cost(y_test, a_test, rule_alert_test, cp) / len(test) * 1000
    c_naive = cost_per_1000(y_test, a_test, s_test, tau, cp)
    c_cost = cost_per_1000(y_test, a_test, sc_test, tau_c, cp)

    print("\n" + "-" * 72)
    print("HEADLINE (held-out test, cost per 1,000 transactions; lower is better)")
    print("-" * 72)
    print(f"  allow everything      : {c_allow:8.2f}")
    print(f"  rules                 : {c_rules:8.2f}")
    print(f"  naive model @ tau={tau:.3f} : {c_naive:8.2f}")
    print(f"  COST-SENSITIVE @ tau={tau_c:.3f}: {c_cost:8.2f}   <-- cost-minimizing policy")

    m = prf(y_test, sc_test >= tau_c)
    print(f"\n  cost-sensitive at tau* on test: precision {m['precision']:.3f}  "
          f"recall {m['recall']:.3f}  (alerts {int((sc_test>=tau_c).sum())})")

    mean_sav, lo, hi, _, _ = bootstrap_savings(
        y_test, a_test, sc_test, rule_alert_test, tau_c, cp, n_boot=2000)
    print(f"\n  => the cost-sensitive threshold SAVES {mean_sav:.2f} per 1,000 "
          f"transactions\n     versus the rules, 95% bootstrap interval "
          f"[{lo:.2f}, {hi:.2f}].")

    # From here on the "model" is the cost-sensitive one.
    s_test, tau, c_model = sc_test, tau_c, c_cost
    s_val = sc_val

    # 3. Does a per-amount-band threshold beat one global threshold?
    bands = [(0.0, 50.0), (50.0, 200.0), (200.0, np.inf)]
    band_taus = per_band_thresholds(val[IS_FRAUD].to_numpy(), val[AMOUNT].to_numpy(),
                                    s_val, cp, bands)
    band_alert = apply_per_band(s_test, a_test, band_taus)
    c_band = binary_cost(y_test, a_test, band_alert, cp) / len(test) * 1000
    print("\n" + "-" * 72)
    print("Global vs per-amount-band threshold (cost per 1,000 on test)")
    print("-" * 72)
    for (blo, bhi), bt in band_taus.items():
        hi_txt = "inf" if bhi == np.inf else f"{bhi:.0f}"
        print(f"  band [{blo:.0f}, {hi_txt}): tau = {bt:.3f}")
    print(f"  global threshold cost : {c_model:8.2f}")
    print(f"  per-band threshold cost: {c_band:8.2f}")
    rel = (c_model - c_band) / c_model if c_model else 0.0
    if rel > 0.05:
        verdict = f"pays off ({rel:.1%} cheaper)"
    elif rel > 0:
        verdict = f"barely helps ({rel:.1%}) -- not worth the complexity, keep one global tau"
    else:
        verdict = "does NOT help -- keep one global tau"
    print(f"  => per-band threshold {verdict}.")

    # 4. SHAP: top-3 reasons for a few caught frauds (true positives).
    print("\n" + "-" * 72)
    print("SHAP top-3 reasons for sample alerts (caught frauds)")
    print("-" * 72)
    contribs, _ = shap_contributions(det_c, test[ALL_FEATURES].to_numpy(), ALL_FEATURES)
    tp_idx = np.where((s_test >= tau) & (y_test == 1))[0][:4]
    for i in tp_idx:
        reasons = top_reasons(contribs[i], ALL_FEATURES, k=3)
        pat = test.iloc[i].get("fraud_pattern", "")
        rtxt = ", ".join(f"{name} (+{v:.2f})" for name, v in reasons)
        print(f"  txn#{i} score={s_test[i]:.3f} amount={a_test[i]:.0f} "
              f"[{pat}]\n     -> {rtxt}")


if __name__ == "__main__":
    main()
