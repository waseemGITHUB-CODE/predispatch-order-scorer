"""HTTP API.

Deliberately thin. The model and its evaluation are the project; this is the
surface that makes them usable. Three endpoints: score an order, read the
held-out evaluation, and pull real examples from the test set so the demo scores
orders that actually happened rather than ones invented to flatter it.

Run:  uvicorn predispatch.api:app --reload
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from predispatch.actions import break_even_risk, decide, explain, load_policy
from predispatch.config import ARTIFACTS, PROCESSED
from predispatch.locale import as_options
from predispatch.features import ALL_FEATURES

app = FastAPI(
    title="Pre-dispatch order-failure scorer",
    description=(
        "Scores an order before it ships for the risk of not being delivered as "
        "promised — cancelled, never delivered, or late. Trained and measured on "
        "~100k real orders from the Olist dataset, with a chronological held-out "
        "test set."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@lru_cache
def _model():
    path = ARTIFACTS / "model.joblib"
    if not path.exists():
        raise HTTPException(503, "No model. Run `python -m predispatch.train`.")
    return joblib.load(path)


@lru_cache
def _metrics() -> dict:
    path = ARTIFACTS / "metrics.json"
    if not path.exists():
        raise HTTPException(503, "No metrics. Run `python -m predispatch.train`.")
    return json.loads(path.read_text())


@lru_cache
def _baseline_row() -> pd.DataFrame:
    """The typical order that explanations are measured against."""
    path = ARTIFACTS / "baseline_row.json"
    if not path.exists():
        raise HTTPException(503, "No baseline row. Run `python -m predispatch.train`.")
    return pd.read_json(path, orient="records")[ALL_FEATURES]


@lru_cache
def _examples() -> pd.DataFrame:
    path = PROCESSED / "test_scored.parquet"
    if not path.exists():
        raise HTTPException(503, "No scored test set. Run `python -m predispatch.train`.")
    return pd.read_parquet(path)


class Order(BaseModel):
    """An order as known at the moment of dispatch.

    Every field is information a merchant genuinely holds before shipping.
    Nothing here is recorded after the outcome — that is the whole point.
    """

    price_total: float = Field(..., ge=0, description="Item value, R$")
    freight_total: float = Field(..., ge=0, description="Shipping charged, R$")
    n_items: int = Field(1, ge=1)
    n_distinct_products: int = Field(1, ge=1)
    n_sellers: int = Field(1, ge=1)
    product_weight_g: Optional[float] = None
    product_length_cm: Optional[float] = None
    product_height_cm: Optional[float] = None
    product_width_cm: Optional[float] = None
    product_photos_qty: Optional[float] = None
    payment_value: Optional[float] = None
    max_installments: int = 1
    n_payment_methods: int = 1
    distance_km: Optional[float] = Field(None, description="Seller to customer")
    promised_days: int = Field(..., description="Days quoted to the customer")
    handling_days: Optional[int] = Field(None, description="Days to hand to carrier")
    purchase_hour: int = Field(12, ge=0, le=23)
    purchase_dayofweek: int = Field(2, ge=0, le=6)
    purchase_month: int = Field(6, ge=1, le=12)
    payment_type: str = "credit_card"
    customer_state: str = "SP"
    seller_state: str = "SP"
    product_category: str = "unknown"

    def to_row(self) -> dict:
        d = self.model_dump()
        d["price_max"] = d["price_total"] / max(d["n_items"], 1)
        d["is_weekend"] = int(d["purchase_dayofweek"] >= 5)
        d["same_state"] = int(d["customer_state"] == d["seller_state"])
        d["freight_ratio"] = (
            d["freight_total"] / d["price_total"] if d["price_total"] else None
        )
        vol = [d["product_length_cm"], d["product_height_cm"], d["product_width_cm"]]
        d["product_volume_cm3"] = None if any(v is None for v in vol) else vol[0] * vol[1] * vol[2]
        d.setdefault("payment_value", d["price_total"] + d["freight_total"])
        if d["payment_value"] is None:
            d["payment_value"] = d["price_total"] + d["freight_total"]
        return d


class Scored(BaseModel):
    risk: float
    action: str
    action_label: str
    rationale: str
    reasons: list[str]
    policy: dict
    expected_failure_cost: float
    # The risk at which acting on THIS order breaks even. Comparing it to
    # `risk` is the whole decision, and showing both makes the call auditable.
    break_even_risk: Optional[float] = None


STATIC = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui() -> str:
    """The single page: score an order, and read the held-out evidence."""
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_trained": (ARTIFACTS / "model.joblib").exists(),
        "metrics_available": (ARTIFACTS / "metrics.json").exists(),
    }


@app.post("/score", response_model=Scored)
def score(order: Order) -> Scored:
    """Score one order and recommend an action."""
    row = order.to_row()
    X = pd.DataFrame([{c: row.get(c) for c in ALL_FEATURES}])
    model = _model()
    risk = float(model.predict_proba(X)[:, 1][0])

    policy = load_policy()
    action = decide(risk, order.freight_total, policy)
    # What this order is expected to cost if shipped and it fails: freight out
    # and back, weighted by the risk.
    loss = order.freight_total * (1.0 + policy["return_multiplier"])
    expected = risk * loss

    return Scored(
        risk=round(risk, 4),
        action=action.code,
        action_label=action.label,
        rationale=action.rationale,
        reasons=explain(model, X, _baseline_row()),
        policy=policy,
        expected_failure_cost=round(expected, 2),
        break_even_risk=break_even_risk(order.freight_total, policy),
    )


@app.get("/metrics")
def metrics() -> dict:
    """The held-out evaluation: precision, recall, cost curve, baselines."""
    return _metrics()


@app.get("/backtest")
def backtest() -> dict:
    """The rolling backtest, if it has been run.

    Optional rather than required: it is a separate, slower entry point, so the
    UI degrades to hiding its card rather than erroring when the file is absent.
    """
    path = ARTIFACTS / "backtest.json"
    if not path.exists():
        raise HTTPException(404, "No backtest. Run `python -m predispatch.backtest`.")
    return json.loads(path.read_text())


@app.get("/states")
def states() -> list[dict]:
    """Readable labels for the Brazilian state codes the model expects.

    The UI renders `label` and posts `code`, so what a reader sees and what the
    model receives are decoupled on purpose — see `predispatch.locale`.
    """
    return as_options()


@app.get("/cloud")
def cloud(n: int = 2600, seed: int = 7) -> dict:
    """A sample of held-out orders, for the 3D scatter on the front page.

    Returned as parallel arrays rather than a list of objects: at this row count
    that is roughly a third of the JSON, and the client wants columns anyway.

    Every point is a real held-out order with its real outcome, so the picture is
    the evaluation rather than an illustration of it.
    """
    df = _examples()
    if len(df) > n:
        df = df.sample(n=n, random_state=seed)
    df = df.sort_values("predicted_risk")

    a = _metrics()["cost_assumptions"]
    return {
        "freight": [round(float(v), 2) for v in df.freight_total.fillna(0)],
        "risk": [round(float(v), 4) for v in df.predicted_risk],
        "failed": [int(v) for v in df.target],
        "distance": [None if pd.isna(v) else round(float(v)) for v in df.distance_km],
        "n_total": int(len(_examples())),
        "n_shown": int(len(df)),
        # The client draws the break-even sheet from these, so it cannot drift
        # away from the rule the server actually applies.
        "friction": float(a["assumed"]["false_positive_friction_brl"]),
        "prevention": float(a["assumed"]["prevention_rate"]),
        "return_multiplier": float(a["measured"]["return_freight_multiplier"]),
    }


@app.get("/examples")
def examples(n: int = 12, band: str = "all") -> list[dict]:
    """Real orders from the held-out test set, for the demo.

    Scoring orders that actually happened — and whose true outcome is known —
    keeps the demo honest. `band` filters to high or low predicted risk so both
    ends are easy to show.
    """
    df = _examples()
    t = load_policy()
    if band == "high":
        df = df[df.predicted_risk >= t["confirm_at"]]
    elif band == "low":
        df = df[df.predicted_risk < t["confirm_at"]]

    cols = ALL_FEATURES + ["predicted_risk", "target", "order_id"]
    out = df.sort_values("predicted_risk", ascending=(band == "low")).head(n)[cols]
    return json.loads(out.to_json(orient="records"))
