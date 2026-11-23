"""Save the Phase 3 result chart: expected cost per 1,000 transactions as a
function of the alert threshold tau, with the cost-minimizing tau* marked and
the rules' fixed cost as a reference line.

    python src/make_chart_phase3.py  ->  reports/phase3_cost_curve.png
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from costs import CostParams, binary_cost, optimal_threshold, sweep
from features import ALL_FEATURES, MerchantRisk, add_causal_features
from generator import GenConfig, generate
from models import XGBDetector
from rules import apply_rules
from run_phase3 import three_way_split
from schema import AMOUNT, IS_FRAUD

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "reports" / "phase3_cost_curve.png"


def main() -> None:
    cp = CostParams(review_fee=4.0, friction=1.0)
    df = add_causal_features(generate(GenConfig(seed=7)))
    train, val, test = three_way_split(df)
    mr = MerchantRisk().fit(train)
    for f in (train, val, test):
        f["merchant_risk"] = mr.transform(f)

    Xtr, ytr, a_tr = (train[ALL_FEATURES].to_numpy(), train[IS_FRAUD].to_numpy(),
                      train[AMOUNT].to_numpy())
    det = XGBDetector().fit(Xtr, ytr, sample_weight=np.where(ytr == 1, a_tr, cp.c_review))

    s_val = det.score(val[ALL_FEATURES].to_numpy())
    s_test = det.score(test[ALL_FEATURES].to_numpy())
    y_test, a_test = test[IS_FRAUD].to_numpy(), test[AMOUNT].to_numpy()

    tau = optimal_threshold(val[IS_FRAUD].to_numpy(), val[AMOUNT].to_numpy(), s_val, cp)
    taus, costs = sweep(y_test, a_test, s_test, cp, n_grid=300)
    c_at_tau = costs[np.argmin(np.abs(taus - tau))]
    c_rules = binary_cost(y_test, a_test, apply_rules(test)["rule_alert"].to_numpy(),
                          cp) / len(test) * 1000

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(taus, costs, color="#4C78A8", lw=2, label="model: expected cost(τ)")
    ax.axvline(tau, color="#E45756", ls="--", lw=1.5,
               label=f"τ* (chosen on validation) = {tau:.3f}")
    ax.scatter([tau], [c_at_tau], color="#E45756", zorder=5, s=55)
    ax.axhline(c_rules, color="grey", ls=":", lw=1.5,
               label=f"rules (fixed) = {c_rules:.0f}")
    ax.annotate(f"{c_at_tau:.0f}", (tau, c_at_tau),
                textcoords="offset points", xytext=(8, 8), color="#E45756")
    ax.set_xlabel("alert threshold τ")
    ax.set_ylabel("expected cost per 1,000 transactions")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_title("Phase 3 — cost-minimizing threshold (held-out test)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
