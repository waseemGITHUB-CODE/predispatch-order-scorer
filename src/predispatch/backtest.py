"""Repeat the whole experiment at several points in time.

Run:  python -m predispatch.backtest

The headline finding — that a tuned threshold makes money on one window and
loses it on the next — rests on a single boundary. One boundary is one
observation. If mid-2018 happened to be an unusual stretch in Brazil, the
finding could be an artefact of where the knife fell rather than anything
about thresholds.

So this cuts the history at five points, and at each one repeats everything:
fit on the past, choose the operating threshold on a validation slice inside
that past, then score the window that follows. Nothing downstream of a cut is
ever visible to anything fitted before it.

Deliberately a separate entry point. It costs a few minutes and two model fits
per window, and `train.py` should stay quick enough to run casually.

The result is allowed to be unflattering. If the per-order rule wins in three
windows out of five rather than five out of five, that is the finding, and a
panel discovering it before we did would be far worse than reporting it.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

from predispatch.config import ARTIFACTS
from predispatch.cost import (
    CostAssumptions,
    cost_curve,
    evaluate_policy,
    evaluate_threshold,
    expected_value_flags,
)
from predispatch.data import build_order_table
from predispatch.features import build_features, feature_matrix
from predispatch.train import build_pipeline

N_WINDOWS = 5
# Sized so the five test windows are strictly disjoint. Cuts are evenly spaced
# between FIRST_CUT and (1 - TEST_FRACTION), giving a stride of
# (1 - TEST_FRACTION - FIRST_CUT) / (N_WINDOWS - 1) = 11.25% of history. A test
# window wider than that stride would overlap its neighbour, and five
# overlapping windows are not five observations — the first version used 0.12
# and shared ~1,200 orders between consecutive windows.
TEST_FRACTION = 0.10
FIRST_CUT = 0.45           # never fit on less than this share of the data
VALIDATION_FRACTION = 0.25  # of each window's own training window


def _fit_and_score(X_tr, y_tr, X_te):
    model = CalibratedClassifierCV(
        build_pipeline(), method="sigmoid", cv=TimeSeriesSplit(n_splits=4)
    ).fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def _threshold_from_validation(train, X_tr, y_tr, a) -> dict:
    """The same honest selection `train.py` does, scoped to this window."""
    cut = int(len(X_tr) * (1 - VALIDATION_FRACTION))
    inner = CalibratedClassifierCV(
        build_pipeline(), method="sigmoid", cv=TimeSeriesSplit(n_splits=3)
    ).fit(X_tr.iloc[:cut], y_tr[:cut])
    p_val = inner.predict_proba(X_tr.iloc[cut:])[:, 1]
    _, best = cost_curve(y_tr[cut:], p_val, train["freight_total"].iloc[cut:], a)
    return {"threshold": best["threshold"], "flag_rate": best["flag_rate"],
            "target_rate": round(float(y_tr[cut:].mean()), 4)}


def run(n_windows: int = N_WINDOWS) -> dict:
    a = CostAssumptions()
    df = build_features(build_order_table()).sort_values("order_purchase_timestamp")
    df = df.reset_index(drop=True)
    N = len(df)
    test_n = int(N * TEST_FRACTION)

    last_cut = N - test_n
    first_cut = int(N * FIRST_CUT)
    cuts = np.linspace(first_cut, last_cut, n_windows).astype(int)

    windows = []
    for i, cut in enumerate(cuts, start=1):
        t0 = time.perf_counter()
        train, test = df.iloc[:cut], df.iloc[cut:cut + test_n]
        X_tr, y_tr = feature_matrix(train), train.target.to_numpy()
        X_te, y_te = feature_matrix(test), test.target.to_numpy()
        freight = test["freight_total"]

        sel = _threshold_from_validation(train, X_tr, y_tr, a)
        p = _fit_and_score(X_tr, y_tr, X_te)

        ev = evaluate_policy(y_te, expected_value_flags(p, freight, a), freight, a)
        th = evaluate_threshold(y_te, p, freight, sel["threshold"], a)
        q = float(np.quantile(p, 1.0 - sel["flag_rate"]))
        fx = evaluate_policy(y_te, p >= q, freight, a)

        windows.append({
            "window": i,
            "train_n": int(len(train)), "test_n": int(len(test)),
            "test_from": str(test.order_purchase_timestamp.iloc[0])[:10],
            "test_to": str(test.order_purchase_timestamp.iloc[-1])[:10],
            "train_target_rate": round(float(y_tr.mean()), 4),
            "validation_target_rate": sel["target_rate"],
            "test_target_rate": round(float(y_te.mean()), 4),
            "pr_auc": round(float(average_precision_score(y_te, p)), 4),
            "chance": round(float(y_te.mean()), 4),
            "chosen_threshold": sel["threshold"],
            "per_order": _slim(ev),
            "tuned_threshold": _slim(th),
            "fixed_flag_rate": _slim(fx),
            "seconds": round(time.perf_counter() - t0, 1),
        })
        _print_window(windows[-1])

    return {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {"n_windows": n_windows, "test_fraction": TEST_FRACTION,
                       "first_cut_fraction": FIRST_CUT,
                       "scheme": "expanding train, fixed-size disjoint test windows"},
            "windows": windows,
            "summary": _summarise(windows)}


def _slim(r: dict) -> dict:
    return {k: r[k] for k in
            ("flag_rate", "precision", "recall", "net_saving", "net_saving_per_1k_orders")}


def _summarise(w: list[dict]) -> dict:
    ev = [x["per_order"]["net_saving_per_1k_orders"] for x in w]
    th = [x["tuned_threshold"]["net_saving_per_1k_orders"] for x in w]
    fx = [x["fixed_flag_rate"]["net_saving_per_1k_orders"] for x in w]
    beats = sum(1 for i in range(len(w)) if ev[i] > th[i] and ev[i] > fx[i])
    return {
        "n_windows": len(w),
        "per_order_profitable_in": sum(1 for v in ev if v > 0),
        "tuned_threshold_profitable_in": sum(1 for v in th if v > 0),
        "fixed_flag_rate_profitable_in": sum(1 for v in fx if v > 0),
        "per_order_beats_both_in": beats,
        "per_order_median_per_1k": round(float(np.median(ev)), 2),
        "tuned_threshold_median_per_1k": round(float(np.median(th)), 2),
        "per_order_range_per_1k": [round(min(ev), 2), round(max(ev), 2)],
        "test_target_rate_range": [min(x["test_target_rate"] for x in w),
                                   max(x["test_target_rate"] for x in w)],
        "pr_auc_range": [min(x["pr_auc"] for x in w), max(x["pr_auc"] for x in w)],
    }


def _print_window(x: dict) -> None:
    print(f"  window {x['window']}  {x['test_from']} to {x['test_to']}  "
          f"train {x['train_n']:,} -> test {x['test_n']:,}  "
          f"base {x['train_target_rate']:.1%} -> {x['test_target_rate']:.1%}  "
          f"({x['seconds']}s)")
    for label, key in (("per-order", "per_order"), ("threshold", "tuned_threshold"),
                       ("flag rate", "fixed_flag_rate")):
        r = x[key]
        print(f"      {label:12}{r['precision']:>7.3f}{r['recall']:>8.3f}"
              f"{r['flag_rate']:>9.1%}{r['net_saving_per_1k_orders']:>10,.0f} /1k")


def main() -> dict:
    print("=" * 72)
    print("  ROLLING BACKTEST — the experiment repeated at five points in time")
    print("=" * 72)
    out = run()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "backtest.json").write_text(json.dumps(out, indent=2))

    s = out["summary"]
    print()
    print(f"  per-order profitable in      : {s['per_order_profitable_in']}/{s['n_windows']} windows")
    print(f"  tuned threshold profitable in: {s['tuned_threshold_profitable_in']}/{s['n_windows']}")
    print(f"  fixed flag rate profitable in: {s['fixed_flag_rate_profitable_in']}/{s['n_windows']}")
    print(f"  per-order beats both in      : {s['per_order_beats_both_in']}/{s['n_windows']}")
    print(f"  per-order median             : R$ {s['per_order_median_per_1k']:,.0f} /1k"
          f"   range {s['per_order_range_per_1k']}")
    print(f"  threshold median             : R$ {s['tuned_threshold_median_per_1k']:,.0f} /1k")
    print(f"  base rate across windows     : {s['test_target_rate_range']}")
    print("=" * 72)
    return out


if __name__ == "__main__":
    main()
