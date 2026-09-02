"""Load the Olist tables and assemble one row per order, with the target.

The dataset ships as nine normalised CSVs. This module joins them into a single
order-level table and attaches the label. It does no feature engineering — that
belongs in `features.py`, where the leakage rules are enforced.

**The target: "not delivered as promised."**

A merchant quotes the customer a delivery date at checkout. The order is a
failure if that promise is not kept, which happens three ways:

  * the order was cancelled or the item turned out to be unavailable,
  * it was never delivered at all and the promised date has passed,
  * it was delivered, but after the date promised.

These are one loss class, not three: in every case the merchant took the money,
committed fulfilment, and did not deliver what was promised. Roughly 10.9% of
orders fail this way.

**Censoring was checked, not assumed.** An order still in transit when the data
ends has an unknown outcome and must not be labelled a failure. In this dataset
every open order's promised date is already in the past at the cutoff, so there
is nothing to exclude — but the check runs anyway, because that is a property of
this extract rather than a guarantee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from predispatch.config import RAW

# Statuses meaning the order never reached the customer and never will.
_FAILED_STATUSES = ("canceled", "unavailable")
# Statuses meaning the order is still somewhere in the pipeline.
_OPEN_STATUSES = ("shipped", "invoiced", "processing", "created", "approved")


def _read(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(RAW / f"{name}.csv", **kw)


def load_orders() -> pd.DataFrame:
    return _read(
        "olist_orders_dataset",
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )


def load_zip_centroids() -> pd.DataFrame:
    """One lat/long per zip prefix.

    The geolocation table holds a million rows — many points per prefix. Taking
    the median rather than the mean keeps a stray mis-geocoded point from
    dragging a whole prefix across the country.
    """
    geo = _read("olist_geolocation_dataset")
    return (
        geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .median()
        .rename(columns={"geolocation_lat": "lat", "geolocation_lng": "lng"})
        .reset_index()
        .rename(columns={"geolocation_zip_code_prefix": "zip_prefix"})
    )


def _haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance. Brazil is wide enough that flat approximations
    misjudge long hauls, which are exactly the deliveries most at risk."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lng2 - lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _order_item_aggregates() -> pd.DataFrame:
    """Collapse line items to one row per order, joining product attributes.

    A multi-item order ships as one parcel from (usually) one seller, so the
    order — not the line — is the unit at risk.
    """
    items = _read("olist_order_items_dataset", parse_dates=["shipping_limit_date"])
    products = _read("olist_products_dataset")
    sellers = _read("olist_sellers_dataset")

    items = items.merge(products, on="product_id", how="left").merge(
        sellers, on="seller_id", how="left"
    )

    agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_distinct_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        price_total=("price", "sum"),
        price_max=("price", "max"),
        freight_total=("freight_value", "sum"),
        product_weight_g=("product_weight_g", "max"),
        product_length_cm=("product_length_cm", "max"),
        product_height_cm=("product_height_cm", "max"),
        product_width_cm=("product_width_cm", "max"),
        product_photos_qty=("product_photos_qty", "mean"),
        shipping_limit_date=("shipping_limit_date", "min"),
        seller_id=("seller_id", "first"),
        seller_state=("seller_state", "first"),
        seller_zip_prefix=("seller_zip_code_prefix", "first"),
        product_category=("product_category_name", "first"),
    )
    return agg.reset_index()


def _payment_aggregates() -> pd.DataFrame:
    """One row per order. `payment_type` is the *dominant* method by value —
    split payments exist, and the largest share is what characterises the order."""
    pay = _read("olist_order_payments_dataset")
    totals = pay.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        n_payment_methods=("payment_sequential", "count"),
        max_installments=("payment_installments", "max"),
    )
    dominant = (
        pay.sort_values("payment_value", ascending=False)
        .groupby("order_id")["payment_type"]
        .first()
        .rename("payment_type")
    )
    return totals.join(dominant).reset_index()


def build_order_table() -> pd.DataFrame:
    """The joined, order-level table with the label attached."""
    orders = load_orders()
    customers = _read("olist_customers_dataset")
    zips = load_zip_centroids()

    df = orders.merge(customers, on="customer_id", how="left")
    df = df.merge(_order_item_aggregates(), on="order_id", how="left")
    df = df.merge(_payment_aggregates(), on="order_id", how="left")

    # Distance between the seller and the customer, via zip-prefix centroids.
    df = df.merge(
        zips.rename(columns={"zip_prefix": "customer_zip_code_prefix",
                             "lat": "cust_lat", "lng": "cust_lng"}),
        on="customer_zip_code_prefix", how="left",
    )
    df = df.merge(
        zips.rename(columns={"zip_prefix": "seller_zip_prefix",
                             "lat": "sell_lat", "lng": "sell_lng"}),
        on="seller_zip_prefix", how="left",
    )
    df["distance_km"] = _haversine_km(
        df.sell_lat, df.sell_lng, df.cust_lat, df.cust_lng
    )

    df = drop_orders_without_items(df)
    df = attach_target(df)
    return df.sort_values("order_purchase_timestamp").reset_index(drop=True)


def drop_orders_without_items(df: pd.DataFrame) -> pd.DataFrame:
    """Remove orders that have no row in `order_items`. This is leakage.

    775 orders in the raw data have an empty basket, and **every one of them is
    a failure** — the items table is only written when an order is actually
    picked and dispatched, so an empty basket is a *consequence* of the order
    being cancelled or found unavailable, not a fact known beforehand.

    Left in, the model finds it instantly. It was found here by the null values
    it leaves behind: on the first run every one of the fifty highest-risk test
    orders had a null price, and the 106 orders the model flagged at threshold
    0.5 were exactly the 106 empty baskets — producing a *precision of 1.000*
    that was pure artefact.

    None of these columns are on the banned list, because the leak is not in a
    column: it is in a row's absence. That is why this is a filter and not an
    audit rule, and why it is worth writing down — a ban list only catches the
    leaks you already thought of.

    Dropping them is also the honest population definition. An order with
    nothing in it is not an order anyone is deciding whether to dispatch, so it
    was never a member of the population this scorer is for.
    """
    has_items = df.price_total.notna()
    return df.loc[has_items].copy()


def attach_target(df: pd.DataFrame) -> pd.DataFrame:
    """Label each order, and drop any whose outcome is genuinely unknown.

    Returns the frame with `target` plus its three components, kept for the
    breakdown in the README. `features.py` strips them before training.
    """
    cutoff = df.order_purchase_timestamp.max()
    is_open = df.order_status.isin(_OPEN_STATUSES)

    # Still in flight and not yet due: the outcome is unknown, so labelling it
    # either way would be inventing data.
    censored = is_open & (df.order_estimated_delivery_date > cutoff)
    df = df.loc[~censored].copy()
    is_open = df.order_status.isin(_OPEN_STATUSES)

    df["cancelled_or_unavailable"] = df.order_status.isin(_FAILED_STATUSES)
    df["never_delivered"] = is_open & (df.order_estimated_delivery_date <= cutoff)
    df["delivered_late"] = (
        df.order_delivered_customer_date > df.order_estimated_delivery_date
    ).fillna(False)

    df["target"] = (
        df.cancelled_or_unavailable | df.never_delivered | df.delivered_late
    ).astype(int)
    return df


def time_split(df: pd.DataFrame, test_fraction: float):
    """Split chronologically: train on the past, test on the future.

    A random split would let the model learn from orders placed after the ones
    it is judged on — festive spikes, a carrier's bad month, a seller's decline
    all bleed backwards. The reported numbers would be optimistic and would not
    survive deployment.
    """
    df = df.sort_values("order_purchase_timestamp")
    cut = int(len(df) * (1 - test_fraction))
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    boundary = train.order_purchase_timestamp.max()
    assert test.order_purchase_timestamp.min() >= boundary, "time split overlaps"
    return train, test, boundary
