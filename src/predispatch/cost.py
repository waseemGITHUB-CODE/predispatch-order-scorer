"""What being wrong actually costs, in money.

Accuracy is not the objective; money is. This module converts predictions into
a net saving against the honest baseline of *doing nothing*, which is what a
merchant does today.

The two errors are not symmetric, and that asymmetry is the whole reason the
default 0.5 threshold is wrong for this problem:

  * **False negative** — a doomed order ships. The freight is spent going out and
    again coming back. `freight_value` is a real column, so this side is
    **measured**.
  * **False positive** — a good order is held for a confirmation call or nudged
    to prepay. That costs a little operational effort and sometimes the sale.
    Nothing in the dataset measures it, so it is **assumed** and swept.

Every assumption is named in `CostAssumptions` and echoed into the metrics
artifact, so a reader can see exactly which numbers came from data and which
came from a judgement call.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from predispatch.config import (
    FALSE_POSITIVE_FRICTION_BRL,
    RETURN_FREIGHT_MULTIPLIER,
)


@dataclass(frozen=True)
class CostAssumptions:
    """The inputs to the cost model, split by whether they are measured."""

    # measured — from the freight_value column
    return_freight_multiplier: float = RETURN_FREIGHT_MULTIPLIER
    # assumed
    false_positive_friction: float = FALSE_POSITIVE_FRICTION_BRL
    # assumed: intervening (a confirmation call, a prepay nudge) does not save
    # every doomed order. Some customers still refuse; some parcels are late
    # regardless. 0.7 is deliberately conservative rather than flattering.
    prevention_rate: float = 0.70

    def describe(self) -> dict:
        return {
            "measured": {
                "failure_cost": "freight_value x (1 + return_freight_multiplier)",
                "return_freight_multiplier": self.return_freight_multiplier,
            },
            "assumed": {
                "false_positive_friction_brl": self.false_positive_friction,
                "prevention_rate": self.prevention_rate,
            },
        }


def failure_cost(freight: pd.Series, a: CostAssumptions) -> pd.Series:
    """Cost of shipping an order that fails: freight out, then freight back."""
    return freight.fillna(freight.median()) * (1.0 + a.return_freight_multiplier)


def evaluate_policy(
    y_true: np.ndarray,
    flagged: np.ndarray,
    freight: pd.Series,
    a: CostAssumptions,
) -> dict:
    """Net saving of any flagging decision, against doing nothing.

    Takes the decision as a boolean mask rather than a score and a threshold,
    so a global cutoff, a fixed flag rate and a per-order expected-value rule
    can all be measured on exactly the same accounting.
    """
    loss = failure_cost(freight, a).to_numpy()
    y = y_true.astype(bool)

    # Today: every failure ships and costs its freight both ways.
    do_nothing = loss[y].sum()

    # With the model: we pay friction on everything we flag; flagged failures are
    # mostly prevented; unflagged failures cost full price.
    friction = flagged.sum() * a.false_positive_friction
    prevented = loss[y & flagged].sum() * a.prevention_rate
    still_lost = loss[y & ~flagged].sum() + loss[y & flagged].sum() * (1 - a.prevention_rate)
    with_model = friction + still_lost

    tp = int((y & flagged).sum())
    fp = int((~y & flagged).sum())
    fn = int((y & ~flagged).sum())
    tn = int((~y & ~flagged).sum())

    return {
        "flagged": int(flagged.sum()),
        "flag_rate": round(float(flagged.mean()), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "do_nothing_cost": round(float(do_nothing), 2),
        "cost_with_model": round(float(with_model), 2),
        "net_saving": round(float(do_nothing - with_model), 2),
        "net_saving_per_1k_orders": round(float((do_nothing - with_model) / len(y) * 1000), 2),
        # What the false positives cost on their own — the brief asks for this
        # explicitly, and it is the number that stops recall being chased blindly.
        "false_positive_cost": round(float(fp * a.false_positive_friction), 2),
    }


def evaluate_threshold(
    y_true: np.ndarray,
    p: np.ndarray,
    freight: pd.Series,
    threshold: float,
    a: CostAssumptions,
) -> dict:
    """Net saving at one global probability cutoff."""
    r = evaluate_policy(y_true, p >= threshold, freight, a)
    return {"threshold": round(float(threshold), 4), **r}


def expected_value_flags(
    p: np.ndarray, freight: pd.Series, a: CostAssumptions
) -> np.ndarray:
    """Flag an order when intervening is worth more than it costs — per order.

    A single global threshold quietly assumes every order is worth the same to
    save. They are not: the thing being protected is the freight, and freight
    varies by an order of magnitude across this dataset. Intervening is worth it
    exactly when

        risk x freight x (1 + return multiplier) x prevention rate  >  friction

    which rearranges to a *different* probability cutoff for every order — low
    for a heavy, expensive-to-ship parcel, high for a cheap one. The economics
    were always per-order; a global threshold was the approximation.

    It also needs no tuning, which matters more here than the elegance. The
    tuned threshold is fitted to the base rate of the window it was chosen in,
    and this dataset's base rate falls from 11.0% to 6.6% between the training
    and test windows. A threshold does not survive that drift. This rule has
    nothing in it to drift: it re-derives the cutoff per order from that
    order's own freight.
    """
    benefit = p * failure_cost(freight, a).to_numpy() * a.prevention_rate
    return benefit > a.false_positive_friction


def cost_curve(
    y_true: np.ndarray,
    p: np.ndarray,
    freight: pd.Series,
    a: CostAssumptions | None = None,
    n_points: int = 60,
) -> tuple[list[dict], dict]:
    """Sweep thresholds and return the curve plus the cost-optimal point.

    The optimum is rarely 0.5. Showing where it actually falls — and how much
    money the difference is worth — is the point of the exercise.
    """
    a = a or CostAssumptions()
    lo, hi = float(np.quantile(p, 0.50)), float(np.quantile(p, 0.999))
    grid = np.unique(np.round(np.linspace(lo, hi, n_points), 4))
    curve = [evaluate_threshold(y_true, p, freight, t, a) for t in grid]
    best = max(curve, key=lambda r: r["net_saving"])
    return curve, best
