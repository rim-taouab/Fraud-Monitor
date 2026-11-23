"""Phase 1 entry point: generate data, time-split, run the rule engine on the
held-out test set, and print the baseline numbers every later model must beat.

    python src/run_baseline.py

Phase 1 is "done when" we can state the rule engine's precision and recall on a
held-out, TIME-ORDERED test set. That is exactly what this prints.
"""
from __future__ import annotations

import pandas as pd

from generator import GenConfig, generate
from metrics import prf, recall_by_pattern
from rules import apply_rules
from schema import FRAUD_PATTERN, IS_FRAUD, TIMESTAMP
from split import time_split

pd.set_option("display.width", 100)


def main() -> None:
    df = generate(GenConfig(seed=7))
    train, test, cutoff = time_split(df, test_frac=0.25)

    print("=" * 60)
    print("DATA")
    print("=" * 60)
    print(f"total rows      : {len(df):,}")
    print(f"overall fraud   : {df[IS_FRAUD].mean():.4%}  ({df[IS_FRAUD].sum():,} frauds)")
    print(f"time cutoff     : {cutoff}")
    print(f"train rows      : {len(train):,}  (fraud {train[IS_FRAUD].mean():.4%})")
    print(f"test  rows      : {len(test):,}  (fraud {test[IS_FRAUD].mean():.4%})")

    # Rules are stateless parameters; apply directly to the test set. (Each rule
    # computes its own per-card/per-user history WITHIN the test window, using
    # only past rows -- see rules.py. No fitting on train is needed for Phase 1.)
    res = apply_rules(test)
    y_true = test[IS_FRAUD].to_numpy()

    print("\n" + "=" * 60)
    print("RULE ENGINE on held-out, time-ordered TEST set")
    print("=" * 60)
    for col in ["rule_velocity", "rule_impossible_travel", "rule_amount_spike", "rule_alert"]:
        m = prf(y_true, res[col])
        label = col.replace("rule_", "")
        print(f"{label:>18} | precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
              f"f1 {m['f1']:.3f} | tp {m['tp']:>3} fp {m['fp']:>4} fn {m['fn']:>3}")

    print("\nRecall by fraud pattern (combined rule_alert):")
    rbp = recall_by_pattern(test[FRAUD_PATTERN].reset_index(drop=True),
                            res["rule_alert"].reset_index(drop=True))
    print(rbp.to_string(index=False))

    print("\n--> Phase 1 baseline headline (combined rules):")
    m = prf(y_true, res["rule_alert"])
    print(f"    precision = {m['precision']:.3f}   recall = {m['recall']:.3f}   f1 = {m['f1']:.3f}")


if __name__ == "__main__":
    main()
