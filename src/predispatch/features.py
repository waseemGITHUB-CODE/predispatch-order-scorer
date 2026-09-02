"""Features available at the moment of dispatch — and nothing else.

The governing rule: a feature may only use information a merchant could actually
have when deciding whether to ship. Delivery dates, review scores and the final
order status are all recorded afterwards. Feed any of them in and the model
scores beautifully in evaluation and is worthless in production, because at
scoring time they do not exist.

That failure is quiet — the metrics look *better*, not worse — so `leakage_audit`
turns the rule into an assertion and a test enforces it.

Seller history carries the same hazard in subtler form. A seller's failure rate
computed over the whole dataset embeds their future. `add_seller_history` walks
orders in time order and uses only what had already happened, which is what a
live system would know.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from predispatch.config import LEAKY_COLUMNS

NUMERIC_FEATURES = [
    "price_total",
    "price_max",
    "freight_total",
    "freight_ratio",
    "n_items",
    "n_distinct_products",
    "n_sellers",
    "product_weight_g",
    "product_volume_cm3",
    "product_photos_qty",
    "payment_value",
    "max_installments",
    "n_payment_methods",
    "distance_km",
    "promised_days",
    "handling_days",
    "purchase_hour",
    "purchase_dayofweek",
    "purchase_month",
    "is_weekend",
    "same_state",
]

CATEGORICAL_FEATURES = [
    "payment_type",
    "customer_state",
    "seller_state",
    "product_category",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def leakage_audit(columns) -> None:
    """Fail loudly if anything post-outcome is present in `columns`.

    Cheap to run, and the mistake it catches is one that hides itself: a leaked
    column raises the score, so nothing about the metrics looks wrong.

    Note this audits *what the model receives*, not the source frame. The joined
    table legitimately carries delivery dates and the label — that is where the
    target comes from. The invariant is that none of them survive selection.
    """
    present = set(columns) & set(LEAKY_COLUMNS)
    if present:
        raise ValueError(
            f"Post-outcome columns reached the model: {sorted(present)}. "
            "These are recorded after dispatch and cannot be known at scoring time."
        )


def _prior_rate(df: pd.DataFrame, key: str, prefix: str, base: float) -> pd.DataFrame:
    """Track record for `key`, using only orders that came *before* each one.

    Past failure rates are among the strongest signals here and also the easiest
    to compute wrongly: aggregate over the whole dataset and every row silently
    contains its own outcome. Expanding means shifted by one, within each group
    in time order, keep it honest — the first order for a key correctly knows
    nothing about it.
    """
    grp = df.groupby(key, sort=False)["target"]
    df[f"{prefix}_prior_orders"] = grp.cumcount()
    # A key with no history gets the base rate, not an implied zero — which
    # would read as "this one never fails".
    df[f"{prefix}_prior_failure_rate"] = grp.transform(
        lambda s: s.shift(1).expanding().mean()
    ).fillna(base)
    return df


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """Attach time-safe track records for seller, route, category and customer.

    **Built, measured, and then dropped from the model.** These looked like the
    obvious strong features — a seller's past failure rate ought to predict the
    next one. On a held-out time split they made things *worse*:

        without priors   PR-AUC 0.2606   ROC 0.7454
        with priors      PR-AUC 0.2211   ROC 0.7161

    The likely cause is the base-rate drift in this dataset: failures fall from
    11.8% in the training window to 7.1% in the test window. An expanding mean
    encodes the older, worse regime, so the model calibrates against a world
    that no longer exists by the time it is scored. Target-encoded history is
    fragile in exactly this way, and a random split would have hidden it —
    which is the argument for splitting by time.

    Retained so the experiment can be reproduced, not called by the pipeline.
    """
    df = df.sort_values("order_purchase_timestamp").copy()
    base = float(df["target"].mean())
    df["route"] = df["seller_state"].astype(str) + ">" + df["customer_state"].astype(str)

    df = _prior_rate(df, "seller_id", "seller", base)
    df = _prior_rate(df, "route", "route", base)
    df = _prior_rate(df, "product_category", "category", base)
    df = _prior_rate(df, "customer_unique_id", "customer", base)
    return df


# Kept as the old name so callers that predate the route/category features
# still work; the behaviour is now the full history set.
add_seller_history = add_history


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the model's inputs. Expects `add_seller_history` to have run."""
    out = df.copy()
    ts = out.order_purchase_timestamp

    # How long the merchant promised the customer, and how long the seller gave
    # themselves to hand the parcel over. A tight handling window is a common
    # reason a promise gets missed.
    out["promised_days"] = (out.order_estimated_delivery_date - ts).dt.days
    out["handling_days"] = (out.shipping_limit_date - ts).dt.days

    out["purchase_hour"] = ts.dt.hour
    out["purchase_dayofweek"] = ts.dt.dayofweek
    out["purchase_month"] = ts.dt.month
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    out["product_volume_cm3"] = (
        out.product_length_cm * out.product_height_cm * out.product_width_cm
    )
    # Freight as a share of value: cheap item, expensive shipping usually means
    # an awkward parcel or an awkward route.
    out["freight_ratio"] = out.freight_total / out.price_total.replace(0, np.nan)
    out["same_state"] = (out.customer_state == out.seller_state).astype(int)

    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].fillna("unknown").astype(str)

    return out


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select the model's inputs, then audit what was selected."""
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Expected features missing from the frame: {missing}")
    X = df[ALL_FEATURES].copy()
    leakage_audit(X.columns)
    return X
