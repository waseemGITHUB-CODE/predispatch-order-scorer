"""Turning a risk score into something a merchant can act on.

A probability is not a decision. This module turns one into an instruction, and
every line it draws is read from the evaluation artifact rather than chosen
here — so the API cannot drift away from the rule the reported precision and
recall were measured under.

The decision itself is per-order, which is the evaluation's main finding. What
an intervention is worth depends on the freight it protects, so the risk at
which acting starts paying for itself is different for every order: low for a
heavy parcel, high for a cheap one. A single global threshold assumed they were
all the same, and on this data it lost money.

Deliberately three bands, not two. A single cutoff forces every flagged order
into the same heavy-handed response; in practice a merchant has a cheap option
(confirm the order) and an expensive one (require prepayment), and matching the
response to the risk is where most of the value sits.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from predispatch.config import ARTIFACTS


@dataclass(frozen=True)
class Action:
    code: str
    label: str
    rationale: str


SHIP = Action(
    "SHIP",
    "Ship as normal",
    "Risk is within tolerance. Adding friction here would cost more in lost "
    "conversion than the expected failures are worth.",
)
CONFIRM = Action(
    "CONFIRM",
    "Confirm before dispatch",
    "Elevated risk. An automated confirmation message is cheap and resolves most "
    "of these before the parcel moves.",
)
PREPAY = Action(
    "PREPAY",
    "Request prepayment or hold",
    "High risk. Worth asking the customer to prepay, or holding the order for "
    "review, before committing freight in both directions.",
)


def load_policy() -> dict:
    """Read the decision rule training measured and chose, from `metrics.json`.

    Serving must apply the same rule the evaluation reports, or the published
    precision and recall describe something the API does not do. So the policy
    is read from the artifact rather than restated here.
    """
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    bands, assumed = metrics["action_bands"], metrics["cost_assumptions"]
    return {
        "policy": bands["policy"],
        "confirm_at": float(bands["confirm_at"]),
        "prepay_at": float(bands["prepay_at"]),
        "basis": bands["basis"],
        "friction": float(assumed["assumed"]["false_positive_friction_brl"]),
        "prevention_rate": float(assumed["assumed"]["prevention_rate"]),
        "return_multiplier": float(assumed["measured"]["return_freight_multiplier"]),
    }


# Kept as the old name for callers that only want the two band edges.
load_thresholds = load_policy


def decide(risk: float, freight: float, policy: dict | None = None) -> Action:
    """Ship, confirm, or ask for prepayment.

    Takes freight as well as risk because the winning policy is per-order: what
    intervention is worth depends on how much freight is at stake, so the same
    risk can be worth acting on for one order and not for another. That is the
    central result of the evaluation, not a detail of the serving code.
    """
    p = policy or load_policy()

    if p["policy"] == "per-order expected value":
        loss = (freight or 0.0) * (1.0 + p["return_multiplier"])
        flag = risk * loss * p["prevention_rate"] > p["friction"]
    else:
        flag = risk >= p["confirm_at"]

    if not flag:
        return SHIP
    return PREPAY if risk >= p["prepay_at"] else CONFIRM


def break_even_risk(freight: float, policy: dict) -> float | None:
    """The risk at which intervening on *this* order starts paying for itself.

    Only meaningful under the per-order policy; a global threshold is the same
    number for everyone and this would just restate it.
    """
    if policy["policy"] != "per-order expected value":
        return None
    loss = (freight or 0.0) * (1.0 + policy["return_multiplier"])
    if loss <= 0:
        return None
    return round(policy["friction"] / (loss * policy["prevention_rate"]), 4)


# How each feature reads in a sentence. Anything not named here is reported by
# its column name, which is ugly but never wrong.
_LABELS = {
    "promised_days": "delivery window promised",
    "handling_days": "days to hand the parcel over",
    "distance_km": "seller-to-customer distance",
    "freight_total": "freight charged",
    "freight_ratio": "freight as a share of order value",
    "price_total": "order value",
    "price_max": "priciest item",
    "product_weight_g": "product weight",
    "product_volume_cm3": "product volume",
    "product_photos_qty": "product photos on the listing",
    "payment_value": "amount paid",
    "max_installments": "instalments",
    "n_payment_methods": "payment methods used",
    "n_items": "items",
    "n_distinct_products": "distinct products",
    "n_sellers": "sellers on the order",
    "purchase_hour": "hour of purchase",
    "purchase_dayofweek": "day of week",
    "purchase_month": "month of purchase",
    "is_weekend": "weekend purchase",
    "same_state": "seller and customer in the same state",
    "payment_type": "payment type",
    "customer_state": "customer state",
    "seller_state": "seller state",
    "product_category": "product category",
}


def _describe(feature: str, value) -> str:
    label = _LABELS.get(feature, feature)
    if value is None or (isinstance(value, float) and value != value):
        return f"{label} (not supplied)"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, (int, float)):
        return f"{label} = {value:,.0f}" if abs(value) >= 100 else f"{label} = {value}"
    return f"{label} = {value}"


def explain(model, X_row, baseline_row, top_n: int = 4) -> list[str]:
    """Why *this* order scored what it did, measured rather than asserted.

    Each feature is swapped, one at a time, for its value in a typical order,
    and the order is rescored. The change in risk is that feature's
    contribution here — the model's own arithmetic, not a rule written
    alongside it that can disagree with the score it appears beneath.

    All the variants are scored in a single batched call, so the whole
    explanation costs one `predict_proba`, not one per feature.
    """
    import pandas as pd

    features = list(X_row.columns)
    variants = pd.concat([X_row] * (len(features) + 1), ignore_index=True)
    for i, col in enumerate(features, start=1):
        variants.loc[i, col] = baseline_row.iloc[0][col]

    p = model.predict_proba(variants)[:, 1]
    actual, deltas = p[0], p[0] - p[1:]

    ranked = sorted(zip(features, deltas), key=lambda kv: -abs(kv[1]))
    reasons = []
    for col, d in ranked[:top_n]:
        # A tenth of a percentage point is not a reason anyone should read.
        if abs(d) < 0.001:
            continue
        direction = "raises" if d > 0 else "lowers"
        reasons.append(
            f"{_describe(col, X_row.iloc[0][col])} — {direction} risk by "
            f"{abs(100 * d):.1f} points"
        )
    if not reasons:
        reasons.append(
            "No single factor moves this order much; the score is the "
            "combination of many small ones."
        )
    return reasons
