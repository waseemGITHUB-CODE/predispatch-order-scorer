"""Evaluation beyond a single precision/recall pair.

Four things a headline number hides, and each is a question a risk team asks
before trusting a score:

* **Is it fair to the baselines?** Each rule flags whatever share of orders it
  happens to flag, so comparing them at their natural operating points measures
  aggressiveness as much as skill. `matched_flag_rate_comparison` puts the model
  at the *same* flag rate as each rule, which is the comparison that isolates
  ranking quality.
* **Do the probabilities mean anything?** A score of 0.7 should correspond to
  roughly 70 failures per hundred. `calibration_table` checks that directly.
* **Where does it break?** An average hides the segments it is systematically
  wrong about. `segment_analysis` surfaces them.
* **What is it using?** `permutation_importance_table` measures which features
  the held-out performance actually depends on, rather than what the model
  claims internally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_curve

from predispatch.baselines import BASELINES
from predispatch.cost import CostAssumptions, evaluate_threshold


def threshold_for_flag_rate(p: np.ndarray, rate: float) -> float:
    """The model threshold that flags `rate` of orders."""
    rate = min(max(rate, 1e-6), 1.0)
    return float(np.quantile(p, 1.0 - rate))


def matched_flag_rate_comparison(
    test: pd.DataFrame, y: np.ndarray, p: np.ndarray, freight: pd.Series,
    a: CostAssumptions,
) -> list[dict]:
    """Each rule against the model *at the same flag rate*.

    A rule that flags 60% of orders will always find more failures than a model
    flagging 5%. Holding the flag rate equal removes that advantage and asks the
    only question that matters: given the same budget of interventions, which
    picks the better orders?
    """
    rows = []
    for name, fn in BASELINES.items():
        s = fn(test)
        rule = evaluate_threshold(y, s, freight, 0.5, a)
        t = threshold_for_flag_rate(p, rule["flag_rate"])
        model = evaluate_threshold(y, p, freight, t, a)
        rows.append({
            "baseline": name,
            "flag_rate": rule["flag_rate"],
            "rule_precision": rule["precision"],
            "rule_recall": rule["recall"],
            "rule_net_saving": rule["net_saving"],
            "model_precision": model["precision"],
            "model_recall": model["recall"],
            "model_net_saving": model["net_saving"],
            "precision_lift": round(
                model["precision"] / rule["precision"], 2) if rule["precision"] else None,
        })
    return rows


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Predicted vs observed failure rate, by decile of predicted risk.

    Quantile bins rather than equal-width: predictions cluster near zero, so
    equal-width bins would leave the interesting high-risk buckets nearly empty.
    """
    df = pd.DataFrame({"p": p, "y": y})
    df["bin"] = pd.qcut(df.p.rank(method="first"), n_bins, labels=False)
    out = df.groupby("bin").agg(
        n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")
    ).reset_index()
    return [
        {
            "bin": int(r.bin) + 1,
            "n": int(r.n),
            "predicted_risk": round(float(r.predicted), 4),
            "observed_rate": round(float(r.observed), 4),
            "gap": round(float(r.predicted - r.observed), 4),
        }
        for r in out.itertuples()
    ]


def segment_analysis(
    test: pd.DataFrame, y: np.ndarray, flagged: np.ndarray, min_n: int = 300,
) -> dict:
    """Where the model does well and badly, by segment.

    Takes the flag decision as a mask rather than a threshold, so the breakdown
    describes the policy that actually ships — which is per-order, and has no
    single threshold to pass in.

    Reported so the weak spots are visible rather than averaged away. Segments
    below `min_n` are dropped — a recall figure over forty orders is noise.
    """
    df = test.copy()
    df["y"], df["flagged"] = y, flagged

    def by(col: str) -> list[dict]:
        rows = []
        for value, g in df.groupby(col):
            if len(g) < min_n or g.y.sum() == 0:
                continue
            tp = int((g.y.astype(bool) & g.flagged).sum())
            rows.append({
                "segment": str(value),
                "n": int(len(g)),
                "failure_rate": round(float(g.y.mean()), 4),
                "recall": round(tp / int(g.y.sum()), 4),
                "flag_rate": round(float(g.flagged.mean()), 4),
            })
        return sorted(rows, key=lambda r: r["recall"])

    return {
        "by_customer_state": by("customer_state")[:8],
        "by_payment_type": by("payment_type"),
        "by_same_state": by("same_state"),
    }


def permutation_importance_table(
    model, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 5, seed: int = 42
) -> list[dict]:
    """Which features the held-out score actually depends on.

    Permutation importance on the test set, scored by average precision. It
    measures what the model's *performance* relies on, which is what matters —
    unlike split-count importances, which reward high-cardinality features for
    offering more places to split.
    """
    r = permutation_importance(
        model, X, y, scoring="average_precision",
        n_repeats=n_repeats, random_state=seed, n_jobs=-1,
    )
    rows = [
        {"feature": c, "importance": round(float(m), 5), "std": round(float(s), 5)}
        for c, m, s in zip(X.columns, r.importances_mean, r.importances_std)
    ]
    return sorted(rows, key=lambda d: d["importance"], reverse=True)


def pr_curve_points(y: np.ndarray, p: np.ndarray, n: int = 60) -> list[dict]:
    """Thinned precision-recall curve, small enough to ship in JSON."""
    precision, recall, _ = precision_recall_curve(y, p)
    idx = np.unique(np.linspace(0, len(precision) - 1, n).astype(int))
    return [
        {"recall": round(float(recall[i]), 4), "precision": round(float(precision[i]), 4)}
        for i in idx
    ]


def full_report(model, test, X_test, y, p, freight, flagged, a=None) -> dict:
    a = a or CostAssumptions()
    return {
        "pr_curve": pr_curve_points(y, p),
        "calibration": calibration_table(y, p),
        "matched_flag_rate": matched_flag_rate_comparison(test, y, p, freight, a),
        "segments": segment_analysis(test, y, flagged),
        "permutation_importance": permutation_importance_table(model, X_test, y),
        "pr_auc_check": round(float(average_precision_score(y, p)), 4),
    }
