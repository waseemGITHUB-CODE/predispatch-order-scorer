"""Paths and the constants that define the problem.

Everything that decides *what* we predict and *what being wrong costs* lives
here, so those choices are visible in one place rather than buried in the
pipeline.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

# --- the split -------------------------------------------------------------
# Split by time, never at random. A random split lets the model train on orders
# that happened *after* the ones it is tested on, which is information no live
# system could have. The last 20% of orders by purchase date are held out and
# never touched during training or tuning.
TEST_FRACTION = 0.20

# --- columns that must never reach the model -------------------------------
# Each of these is recorded *after* the order ships, so knowing it at scoring
# time is impossible. They are the difference between a model that works and one
# that merely looks like it does, and `features.leakage_audit` enforces the ban.
LEAKY_COLUMNS = frozenset({
    "order_status",                   # the outcome itself
    "order_delivered_customer_date",  # when it arrived
    "order_delivered_carrier_date",   # when the carrier collected it
    "order_approved_at",              # payment approval, after purchase
    "review_score",                   # written after delivery
    "review_comment_title",
    "review_comment_message",
    "review_creation_date",
    "review_answer_timestamp",
    "delivered_late",                 # target components
    "never_delivered",
    "cancelled_or_unavailable",
    "target",
})

# --- what an error costs ---------------------------------------------------
# A false negative is measured: shipping a doomed order burns the freight out
# and again on the way back. `freight_value` is a real column, so this side of
# the ledger comes from the data.
RETURN_FREIGHT_MULTIPLIER = 1.0   # return leg costs about the same as the outbound

# A false positive is assumed, and stated as such. Flagging a good order adds
# friction — a confirmation call, or a nudge to prepay — which costs a little
# operational effort and occasionally loses the sale. There is nothing in the
# dataset that measures this, so it is a parameter the cost curve is swept over
# rather than a fact.
FALSE_POSITIVE_FRICTION_BRL = 5.0
