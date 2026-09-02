"""How much of the result survives if the assumed costs are wrong.

Two numbers in the cost model are judgement calls rather than measurements:
the friction of bothering a good customer (R$5) and the share of caught
failures an intervention actually prevents (70%). The freight side is real,
taken from the dataset's own column, but these two are mine.

That matters more here than it usually would, because the shipped policy is
*derived* from them:

    flag when   risk x freight x (1 + m) x prevention  >  friction

Change the friction and you change both what gets flagged and what the
flagging is worth. So a sweep has to re-derive the decisions at every point,
not just re-price a fixed set of flags — which is what `_at` does below.

The output is the answer to the obvious hostile question, "what if your R$5 is
wrong?": the range over which the conclusion holds, and the point where it
stops holding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from predispatch.cost import (
    CostAssumptions,
    evaluate_policy,
    expected_value_flags,
)

# Friction values to sweep, in R$ per good order flagged. Spans "a templated
# SMS costs almost nothing" through "every flag needs a human phone call".
FRICTION_GRID = [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]

# Share of caught failures an intervention actually saves. 1.0 is not offered:
# claiming every warned order is rescued would be the flattering assumption
# this whole module exists to avoid.
PREVENTION_GRID = [0.3, 0.5, 0.7, 0.9]


def _at(y_true, p, freight, friction: float, prevention: float, base: CostAssumptions) -> dict:
    """Net saving when the merchant both decides and is scored under these costs.

    Self-consistency is the point: a merchant who believes friction is R$15
    would flag fewer orders *and* be judged against R$15. Holding the flags
    fixed while re-pricing them would measure a policy nobody would run.
    """
    a = CostAssumptions(
        return_freight_multiplier=base.return_freight_multiplier,
        false_positive_friction=friction,
        prevention_rate=prevention,
    )
    r = evaluate_policy(y_true, expected_value_flags(p, freight, a), freight, a)
    return {
        "friction": friction,
        "prevention": prevention,
        "flag_rate": r["flag_rate"],
        "precision": r["precision"],
        "recall": r["recall"],
        "net_saving": r["net_saving"],
        "net_saving_per_1k_orders": r["net_saving_per_1k_orders"],
    }


def friction_breakeven(y_true, p, freight, prevention: float, base: CostAssumptions,
                       lo: float = 0.25, hi: float = 60.0, step: float = 0.25) -> float | None:
    """Highest friction at which the policy is *still reliably* profitable.

    Found by walking a fine grid upward and stopping at the **first** crossing,
    not by bisecting to the last one. That distinction matters, and getting it
    wrong is how this function was first written.

    Net saving is not monotonic in friction. As friction rises the policy flags
    fewer and fewer orders — 18% of the book at R$5, 0.4% at R$30 — and once the
    flag count is small the net saving rattles around zero on the luck of which
    handful of orders remain. Bisection happily locks onto one of those late
    positive flickers and reports a headroom an order of magnitude too generous.
    The first crossing is the honest answer: past it, the result is noise.
    """
    n = int((hi - lo) / step) + 1
    grid = [round(lo + i * step, 2) for i in range(n)]
    if _at(y_true, p, freight, grid[0], prevention, base)["net_saving"] <= 0:
        return None
    last_good = grid[0]
    for f in grid[1:]:
        if _at(y_true, p, freight, f, prevention, base)["net_saving"] <= 0:
            return last_good
        last_good = f
    return last_good


def sensitivity_report(y_true: np.ndarray, p: np.ndarray, freight: pd.Series,
                       base: CostAssumptions | None = None) -> dict:
    """Sweep both assumed inputs and locate where the conclusion breaks."""
    base = base or CostAssumptions()

    grid = [
        _at(y_true, p, freight, f, v, base)
        for v in PREVENTION_GRID
        for f in FRICTION_GRID
    ]
    breakeven = {
        str(v): friction_breakeven(y_true, p, freight, v, base)
        for v in PREVENTION_GRID
    }
    at_base = _at(y_true, p, freight, base.false_positive_friction, base.prevention_rate, base)
    headroom = breakeven.get(str(base.prevention_rate))

    return {
        "shipped_assumptions": {
            "friction": base.false_positive_friction,
            "prevention": base.prevention_rate,
            "net_saving": at_base["net_saving"],
        },
        "grid": grid,
        "friction_breakeven_by_prevention": breakeven,
        # The single number worth quoting: how far the friction assumption can
        # be wrong before the policy stops paying for itself.
        "friction_headroom": headroom,
        "friction_headroom_multiple": (
            round(headroom / base.false_positive_friction, 1)
            if headroom else None
        ),
        "profitable_share_of_grid": round(
            sum(1 for g in grid if g["net_saving"] > 0) / len(grid), 3
        ),
    }
