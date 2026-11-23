"""Save the Phase 2 result chart: precision-recall curves for the three
detectors on the held-out test set, with the rule baseline as a single point.

    python src/make_chart_phase2.py  ->  reports/phase2_model_comparison.png
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from features import ALL_FEATURES, MerchantRisk, add_causal_features
from generator import GenConfig, generate
from metrics import prf
from ml_metrics import pr_auc
from models import DETECTORS
from rules import apply_rules
from schema import IS_FRAUD
from split import time_split

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "reports" / "phase2_model_comparison.png"
COLORS = {"logreg": "#4C78A8", "xgboost": "#E45756", "isolation_forest": "#72B043"}


def main() -> None:
    df = add_causal_features(generate(GenConfig(seed=7)))
    train, test, _ = time_split(df, test_frac=0.25)
    mr = MerchantRisk().fit(train)
    train = train.copy(); test = test.copy()
    train["merchant_risk"] = mr.transform(train)
    test["merchant_risk"] = mr.transform(test)

    Xtr, ytr = train[ALL_FEATURES].to_numpy(), train[IS_FRAUD].to_numpy()
    Xte, yte = test[ALL_FEATURES].to_numpy(), test[IS_FRAUD].to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for Det in DETECTORS:
        det = Det().fit(Xtr, ytr)
        s = det.score(Xte)
        p, r, _ = precision_recall_curve(yte, s)
        ax.plot(r, p, label=f"{det.name} (PR-AUC {pr_auc(yte, s):.3f})",
                color=COLORS[det.name], lw=2)

    rb = prf(yte, apply_rules(test)["rule_alert"].to_numpy())
    ax.scatter([rb["recall"]], [rb["precision"]], color="black", zorder=5, s=70,
               marker="*", label=f"rules (P {rb['precision']:.2f}, R {rb['recall']:.2f})")

    ax.axhline(yte.mean(), ls="--", color="grey", lw=1,
               label=f"random ({yte.mean():.3%} fraud)")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("Phase 2 — precision-recall on held-out, time-ordered test")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
