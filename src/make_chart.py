"""Save the Phase 1 result chart (per the plan's 'one chart per phase' habit).

Left panel : recall per fraud pattern under the rule engine -- shows WHICH
             attacks the rules catch vs. miss (the gap ML must close).
Right panel: the precision/recall/F1 of the combined rule alert, the baseline
             every later model must beat.

    python src/make_chart.py   ->   data/phase1_rule_baseline.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from generator import GenConfig, generate
from metrics import prf, recall_by_pattern
from rules import apply_rules
from schema import FRAUD_PATTERN, IS_FRAUD
from split import time_split

OUT = Path(__file__).resolve().parents[1] / "reports" / "phase1_rule_baseline.png"


def main() -> None:
    df = generate(GenConfig(seed=7))
    _, test, _ = time_split(df, test_frac=0.25)
    res = apply_rules(test)
    y_true = test[IS_FRAUD].to_numpy()

    rbp = recall_by_pattern(test[FRAUD_PATTERN].reset_index(drop=True),
                            res["rule_alert"].reset_index(drop=True))
    m = prf(y_true, res["rule_alert"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.barh(rbp["pattern"], rbp["recall"], color="#4C78A8")
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("recall")
    ax1.set_title("Rule-engine recall by fraud pattern")
    for y, (r, c, t) in enumerate(zip(rbp["recall"], rbp["caught"], rbp["total"])):
        ax1.text(min(r + 0.02, 0.9), y, f"{c}/{t}", va="center", fontsize=9)

    names = ["precision", "recall", "f1"]
    vals = [m["precision"], m["recall"], m["f1"]]
    bars = ax2.bar(names, vals, color=["#54A24B", "#E45756", "#B279A2"])
    ax2.set_ylim(0, 1)
    ax2.set_title("Combined rule alert (held-out, time-ordered)")
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                 ha="center", fontsize=10)

    fig.suptitle("Phase 1 baseline — rules every later model must beat", fontsize=13)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
