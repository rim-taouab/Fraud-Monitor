"""Phase 2 entry point: engineer causal features, train the three detectors,
and print the results table comparing rules / logreg / xgboost / isolation
forest on the SAME held-out, time-ordered test set.

    python src/run_phase2.py

Phase 2 is "done when" that comparison table exists. This prints it, plus a
time-series-CV check (is the test PR-AUC a fluke?) and a calibration (Brier)
reading for the booster.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

from features import ALL_FEATURES, MerchantRisk, add_causal_features
from generator import GenConfig, generate
from metrics import prf
from ml_metrics import pr_auc, recall_at_precision
from models import DETECTORS, XGBDetector
from rules import apply_rules
from schema import IS_FRAUD
from split import time_split

warnings.filterwarnings("ignore")
pd.set_option("display.width", 100)


def _xy(frame):
    return frame[ALL_FEATURES].to_numpy(), frame[IS_FRAUD].to_numpy()


def main() -> None:
    # 1. Data + causal features (safe on the full frame: past-only).
    df = generate(GenConfig(seed=7))
    df = add_causal_features(df)
    train, test, cutoff = time_split(df, test_frac=0.25)

    # 2. The one label-derived feature is fit on TRAIN only (anti-leakage).
    mr = MerchantRisk().fit(train)
    train = train.copy()
    test = test.copy()
    train["merchant_risk"] = mr.transform(train)
    test["merchant_risk"] = mr.transform(test)

    Xtr, ytr = _xy(train)
    Xte, yte = _xy(test)

    print("=" * 72)
    print(f"train {len(train):,} rows (fraud {ytr.mean():.3%}) | "
          f"test {len(test):,} rows (fraud {yte.mean():.3%}) | cutoff {cutoff}")
    print("=" * 72)

    # 3. Baseline: the rule engine on the same test rows.
    rule_res = apply_rules(test)
    rb = prf(yte, rule_res["rule_alert"].to_numpy())

    rows = [{
        "model": "rules (baseline)",
        "PR-AUC": pr_auc(yte, rule_res["rule_alert"].astype(int).to_numpy()),
        "recall@P>=0.50": rb["recall"] if rb["precision"] >= 0.50 else 0.0,
        "recall@P>=0.90": rb["recall"] if rb["precision"] >= 0.90 else 0.0,
    }]

    # 4. Train + score each detector on the identical split.
    scores = {}
    for Det in DETECTORS:
        det = Det().fit(Xtr, ytr)
        s = det.score(Xte)
        scores[det.name] = s
        rows.append({
            "model": det.name,
            "PR-AUC": pr_auc(yte, s),
            "recall@P>=0.50": recall_at_precision(yte, s, 0.50),
            "recall@P>=0.90": recall_at_precision(yte, s, 0.90),
        })

    table = pd.DataFrame(rows).set_index("model")
    print("\nResults on held-out, time-ordered TEST set")
    print("(PR-AUC: area under precision-recall curve; higher is better)\n")
    print(table.round(3).to_string())

    best = table["PR-AUC"].drop("rules (baseline)").idxmax()
    lift = table.loc[best, "PR-AUC"] / max(table.loc["rules (baseline)", "PR-AUC"], 1e-9)
    print(f"\nBest model: {best} (PR-AUC {table.loc[best, 'PR-AUC']:.3f}, "
          f"{lift:.1f}x the rules' PR-AUC)")

    # 5. Is the booster's test PR-AUC a fluke? Time-series CV on TRAIN.
    print("\nTime-series CV (XGBoost, 5 folds on train, PR-AUC):")
    tscv = TimeSeriesSplit(n_splits=5)
    cv = []
    for tr_idx, va_idx in tscv.split(Xtr):
        if ytr[tr_idx].sum() == 0 or ytr[va_idx].sum() == 0:
            continue
        m = XGBDetector().fit(Xtr[tr_idx], ytr[tr_idx])
        cv.append(pr_auc(ytr[va_idx], m.score(Xtr[va_idx])))
    if cv:
        print(f"  folds={len(cv)}  mean={np.mean(cv):.3f}  std={np.std(cv):.3f}  "
              f"[{', '.join(f'{c:.3f}' for c in cv)}]")

    # 6. Calibration: are XGBoost's probabilities trustworthy as probabilities?
    brier = brier_score_loss(yte, scores["xgboost"])
    base_brier = brier_score_loss(yte, np.full_like(yte, yte.mean(), dtype=float))
    print(f"\nXGBoost calibration: Brier={brier:.5f} "
          f"(predict-base-rate baseline={base_brier:.5f}; lower is better)")


if __name__ == "__main__":
    main()
