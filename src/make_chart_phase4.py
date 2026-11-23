"""Save the Phase 4 headline chart: the frozen defense bleeding money round
after round while the retrained defense chokes the attacker off.

Two panels, same x-axis (round):
  top    - cumulative stolen (undetected) value: frozen climbs without bound,
           retrained flattens. This is the "frozen decays / retrained holds" story
           measured in money, which is cleaner and more honest than a per-round
           detection rate (a static model has fixed blind spots, so its detection
           does not monotonically decay -- it simply leaks a constant amount
           every round).
  bottom - stolen value PER round: frozen stays high, retrained falls toward 0 as
           each retrain closes the latest gap.

    python src/make_chart_phase4.py  ->  reports/phase4_arms_race.png
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_phase4 import run_arms_race

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "reports" / "phase4_arms_race.png"
FROZEN = "#E45756"
ADAPT = "#4C78A8"


def main() -> None:
    r = run_arms_race(verbose=False)
    rounds = np.array(r["round"])
    pf = np.array(r["profit_frozen"])
    pa = np.array(r["profit_adaptive"])
    cum_f, cum_a = np.cumsum(pf), np.cumsum(pa)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 7.0), sharex=True)

    ax1.plot(rounds, cum_f, color=FROZEN, lw=2.4, marker="o", ms=4,
             label="frozen model (never retrained)")
    ax1.plot(rounds, cum_a, color=ADAPT, lw=2.4, marker="o", ms=4,
             label="retrained model (adapts each round)")
    ax1.fill_between(rounds, cum_a, cum_f, color=FROZEN, alpha=0.08)
    ax1.set_ylabel("cumulative stolen value")
    ax1.set_ylim(bottom=0)
    cut = 1 - cum_a[-1] / max(cum_f[-1], 1)
    ax1.set_title(f"Phase 4 — arms race: retraining cuts the attacker's haul by "
                  f"{cut:.0%}\n(frozen {cum_f[-1]:,.0f} vs retrained {cum_a[-1]:,.0f})")
    ax1.legend(loc="upper left")

    ax2.plot(rounds, pf, color=FROZEN, lw=2.2, marker="o", ms=4,
             label="frozen: leaks every round")
    ax2.plot(rounds, pa, color=ADAPT, lw=2.2, marker="o", ms=4,
             label="retrained: leak falls to ~0")
    ax2.set_xlabel("round (attacker adapts each round)")
    ax2.set_ylabel("stolen value per round")
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(rounds.min(), rounds.max())
    ax2.legend(loc="upper right")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
