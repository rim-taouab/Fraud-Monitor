"""Transaction schema: the shared contract for the whole project.

Every component reads from this one definition:
  - the synthetic generator (src/generator.py) EMITS these columns
  - the rule engine (src/rules.py) and ML features READ these columns
  - the adversarial attacker (Phase 4) PERTURBS these columns
  - the graph layer (Phase 4) links on card_id / device_id / merchant_id

Design rule: raw, un-derived fields only. Anything computed (rolling counts,
distance, implied speed, ratio-to-median) is a FEATURE, built downstream from
these columns, never stored here. That keeps the data honest and the
leakage boundary (see src/split.py) easy to reason about.

Why these fields and not Kaggle's V1..V28:
  Kaggle is PCA-anonymized, so it cannot support card-level velocity,
  geographic impossible-travel, or an attacker that perturbs real knobs.
  This schema exposes exactly the knobs the rules and the attacker need.
  Kaggle stays a separate Phase-2 benchmark, not an input to this schema.
"""
from __future__ import annotations

from enum import IntEnum


# --- Column names (import these everywhere; never hard-code strings) ---------
TXN_ID = "txn_id"            # unique transaction id (monotonic with time)
TIMESTAMP = "timestamp"      # pandas datetime64[ns], UTC-naive; the ONLY time axis
USER_ID = "user_id"          # the human/account
CARD_ID = "card_id"          # a card belonging to a user (user may have several)
MERCHANT_ID = "merchant_id"  # specific merchant
MERCHANT_CATEGORY = "merchant_category"  # see MerchantCategory below
AMOUNT = "amount"            # float, transaction value in a single currency unit
LAT = "lat"                  # float degrees, where the card was used
LON = "lon"                  # float degrees
DEVICE_ID = "device_id"      # device/terminal used; shared devices => fraud rings
IS_FRAUD = "is_fraud"        # int {0,1}, the GROUND-TRUTH label (never a feature)
FRAUD_PATTERN = "fraud_pattern"  # str tag of which injector produced a fraud, else ""

# Order matters: generator emits in this order, downstream can rely on it.
COLUMNS: list[str] = [
    TXN_ID,
    TIMESTAMP,
    USER_ID,
    CARD_ID,
    MERCHANT_ID,
    MERCHANT_CATEGORY,
    AMOUNT,
    LAT,
    LON,
    DEVICE_ID,
    IS_FRAUD,
    FRAUD_PATTERN,
]

# Columns the model/rules are ALLOWED to see. is_fraud and fraud_pattern are
# labels/metadata and must never be fed to a detector (that would be leakage).
LABEL_COLUMNS: list[str] = [IS_FRAUD, FRAUD_PATTERN]
FEATURE_SOURCE_COLUMNS: list[str] = [c for c in COLUMNS if c not in LABEL_COLUMNS]


class MerchantCategory(IntEnum):
    """Coarse MCC-like buckets. Each carries a different normal spend profile
    and a different baseline risk (used by the generator and merchant-risk
    feature later)."""
    GROCERY = 0
    RESTAURANT = 1
    FUEL = 2
    RETAIL = 3
    TRAVEL = 4        # higher ticket, more geographic spread
    ELECTRONICS = 5   # higher ticket, favored by card-not-present fraud
    ATM = 6           # cash withdrawal
    ONLINE = 7        # card-not-present, location is weak signal


# Fraud pattern tags the generator can inject. The attacker (Phase 4) adds to
# this space by discovering new (amount, timing, location, merchant) combos.
class FraudPattern:
    NONE = ""
    VELOCITY_BURST = "velocity_burst"      # many txns in minutes (stolen card drain)
    IMPOSSIBLE_TRAVEL = "impossible_travel"  # two far-apart uses, no time to travel
    AMOUNT_SPIKE = "amount_spike"          # single charge far above user's norm
    CARD_TESTING = "card_testing"          # many tiny online charges to test a card
