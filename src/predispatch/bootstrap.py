"""Error bars on the headline numbers.

"Precision 8.1%" is one figure from one test set. A different 19,733 orders
from the same period would give a slightly different one, and without knowing
how much slightly means, a reader cannot tell a real difference from noise.

The percentile bootstrap answers that: resample the test set with replacement,
recompute, and read the spread off the resulting distribution. It assumes only
that the test set is representative of the period it came from — no assumption
about the shape of the sampling distribution, which matters here because
precision on a 6.6% positive rate is not remotely normal.

The interval that earns its place most is the last one. We report a matched
flag-rate lift of 0.91x against the tight-promise rule — a loss. Whether that
is a genuine ranking deficit or a coin-flip is a question the point estimate
cannot answer and this can.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from predispatch.baselines import BASELINES
from predispatch.cost import (
    CostAssumptions,
    evaluate_policy,
    expected_value_flags,
)

N_RESAMPLES = 600      # enough for a stable 95% interval; ~30s on this test set
SEED = 20260901


def _ci(vals: list[float], lo: float = 2.5, hi: float = 97.5) -> dict:
    v = np.asarray([x for x in vals if x == x])          # drop any NaN draws
    return {
        "point": None,                                    # filled by the caller
        "lo": round(float(np.percentile(v, lo)), 4),
        "hi": round(float(np.percentile(v, hi)), 4),
        "n_draws": int(len(v)),
    }


def bootstrap_report(test: pd.DataFrame, y: np.ndarray, p: np.ndarray,
                     freight: pd.Series, a: CostAssumptions | None = None,
                     n: int = N_RESAMPLES, seed: int = SEED) -> dict:
    """Percentile intervals for the reported metrics, and for the one loss."""
    a = a or CostAssumptions()
    rng = np.random.default_rng(seed)
    N = len(y)

    freight_arr = freight.to_numpy()
    # The rule we ship, so the interval describes the policy that runs.
    flags_full = expected_value_flags(p, freight, a)

    # The comparison worth testing: our precision minus the rule's, both held
    # at the rule's own flag rate, on the same resample.
    # baselines return Series or ndarray depending on the rule; normalise
    rule_scores = {name: np.asarray(fn(test)) for name, fn in BASELINES.items()}

    draws = {k: [] for k in
             ["precision", "recall", "pr_auc", "net_saving_per_1k", "flag_rate"]}
    lift_draws = {name: [] for name in rule_scores}

    for _ in range(n):
        idx = rng.integers(0, N, N)
        yb, pb, fb = y[idx], p[idx], freight_arr[idx]
        if yb.sum() == 0:
            continue                                      # degenerate resample

        fl = flags_full[idx]
        r = evaluate_policy(yb, fl, pd.Series(fb), a)
        draws["precision"].append(r["precision"])
        draws["recall"].append(r["recall"])
        draws["flag_rate"].append(r["flag_rate"])
        draws["net_saving_per_1k"].append(r["net_saving_per_1k_orders"])
        draws["pr_auc"].append(average_precision_score(yb, pb))

        for name, s in rule_scores.items():
            sb = s[idx]
            rule_flag = sb >= 0.5
            rate = rule_flag.mean()
            if rate <= 0 or rate >= 1:
                continue
            rule_p = yb[rule_flag].mean() if rule_flag.any() else 0.0
            # our model held at exactly that flag rate
            cut = np.quantile(pb, 1.0 - rate)
            ours = pb >= cut
            model_p = yb[ours].mean() if ours.any() else 0.0
            lift_draws[name].append(model_p - rule_p)

    out = {"n_resamples": n, "method": "percentile bootstrap, 95%"}
    for k, v in draws.items():
        out[k] = _ci(v)

    # Precision-difference intervals. An interval straddling zero means the
    # comparison is a tie, not a defeat — which is a materially different claim
    # from the one the point estimate makes on its own.
    out["matched_flag_rate_precision_diff"] = {}
    for name, v in lift_draws.items():
        if len(v) < 50:
            continue
        c = _ci(v)
        c["point"] = round(float(np.mean(v)), 4)
        c["verdict"] = (
            "model better" if c["lo"] > 0 else
            "rule better"  if c["hi"] < 0 else
            "indistinguishable"
        )
        out["matched_flag_rate_precision_diff"][name] = c
    return out
