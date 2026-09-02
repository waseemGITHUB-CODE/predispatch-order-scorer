"""Train the scorer and write the evaluation artifacts.

Run:  python -m predispatch.train

Four methodological choices drive everything here, and each is a place this
kind of project usually goes wrong:

**Split by time, never at random.** Train on the past, test on the future. A
random split lets the model learn from orders placed *after* the ones it is
judged on — a carrier's bad month, a seller's decline, a festive spike — and the
reported numbers would not survive deployment.

**Calibrate by time too.** The usual `CalibratedClassifierCV(cv=k)` reshuffles
the training data, which reintroduces the leak the time split just removed.
`TimeSeriesSplit` calibrates each fold on its own future, so every order
contributes and no fold sees ahead of itself.

**Choose the operating threshold on validation, not on the test set.** Sweeping
thresholds on the test set and reporting the winner is peeking, and it inflates
precision by an amount the reader cannot see. The threshold here comes from the
tail of the training window; `hindsight_best` reports what it would have been
with foresight, so the cost of deciding in advance is visible.

**Compare against simple rules.** A model that barely beats "flag every
cross-state order" is a finding worth reporting, not a result to bury.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from predispatch.baselines import BASELINES
from predispatch.config import ARTIFACTS, PROCESSED, TEST_FRACTION
from predispatch.cost import (
    CostAssumptions,
    cost_curve,
    failure_cost,
    evaluate_policy,
    evaluate_threshold,
    expected_value_flags,
)
from predispatch.bootstrap import bootstrap_report
from predispatch.data import build_order_table, time_split
from predispatch.sensitivity import sensitivity_report
from predispatch.evaluate import full_report
from predispatch.features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_features,
    feature_matrix,
)

def build_pipeline() -> Pipeline:
    """Gradient-boosted trees over mixed numeric and categorical inputs.

    Trees suit this data: the signal is full of interactions (a long distance
    matters more when the promised window is short) and thresholds (freight
    above some share of value), neither of which a linear model captures without
    manual help. `class_weight="balanced"` keeps the ~11% positive class from
    being ignored.

    These settings come from a 72-point grid search scored by average precision
    across four expanding time folds of the *training* window only — the test
    set played no part in choosing them.

    The honest summary of that search is that it barely mattered. Every
    configuration in the top twelve scored between 0.193 and 0.197, against a
    fold-to-fold standard deviation of 0.063: the spread across the whole grid
    is comfortably inside the noise of a single fold. This model is not
    hyperparameter-limited, it is signal-limited, and more tuning would buy
    nothing. The best point is used because it was free, not because it is
    meaningfully better.

    `early_stopping=False` on purpose. The alternative holds out a *random* 10%
    for its stopping signal, which quietly mixes future orders back into a
    pipeline built to keep them out; with `max_iter` already chosen by the
    time-respecting search, it is not needed.
    """
    pre = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    cat_mask = [False] * len(NUMERIC_FEATURES) + [True] * len(CATEGORICAL_FEATURES)
    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.03,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features=cat_mask,
        class_weight="balanced",
        early_stopping=False,
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _choose_threshold(train, X_train, y_train, a, validation_fraction: float = 0.25) -> dict:
    """Pick the operating threshold without ever looking at the test set.

    The obvious shortcut is to sweep thresholds on the test set and report the
    best one. That is peeking: the reported precision and recall then come from
    an operating point that was itself tuned on the labels being reported
    against, and the number flatters the model by an amount nobody can see.

    So the training window is split again, chronologically. A model is fitted on
    the first 75% of it, the cost curve is swept over the last 25%, and the
    threshold that wins there is the one carried forward and applied, unchanged,
    to the test set. The final model is then refitted on the whole training
    window, since throwing away a quarter of the data for good would be a real
    cost paid for nothing.

    The gap between this threshold's test result and `hindsight_best` is
    reported too. That gap is the price of having to choose in advance, and it
    is a number worth showing rather than hiding.
    """
    cut = int(len(X_train) * (1 - validation_fraction))
    fit_X, fit_y = X_train.iloc[:cut], y_train[:cut]
    val_X, val_y = X_train.iloc[cut:], y_train[cut:]
    val_freight = train["freight_total"].iloc[cut:]

    inner = CalibratedClassifierCV(
        build_pipeline(), method="sigmoid", cv=TimeSeriesSplit(n_splits=3)
    ).fit(fit_X, fit_y)
    p_val = inner.predict_proba(val_X)[:, 1]

    _, best = cost_curve(val_y, p_val, val_freight, a)
    return {
        "threshold": best["threshold"],
        "chosen_on": (
            f"the last {validation_fraction:.0%} of the training window "
            f"({len(val_X):,} orders), scored by a model fitted only on what "
            "came before it"
        ),
        "validation_window": [
            str(train.order_purchase_timestamp.iloc[cut]),
            str(train.order_purchase_timestamp.iloc[-1]),
        ],
        "validation_n": int(len(val_X)),
        "validation_target_rate": round(float(val_y.mean()), 4),
        "validation_flag_rate": best["flag_rate"],
        "validation_precision": best["precision"],
        "validation_recall": best["recall"],
        "validation_net_saving": best["net_saving"],
    }


def _typical_row(X_train: pd.DataFrame) -> pd.DataFrame:
    """One unremarkable order: the median of each number, the commonest of each
    category. Serves as the reference point an explanation is measured against.
    """
    row = {}
    for c in NUMERIC_FEATURES:
        row[c] = float(X_train[c].median())
    for c in CATEGORICAL_FEATURES:
        row[c] = X_train[c].mode().iat[0]
    return pd.DataFrame([row])


def _action_bands(p: np.ndarray, freight, a, chosen: dict) -> dict:
    """Where the ship / confirm / prepay lines fall.

    The confirm line is whatever the winning policy decides, so the API applies
    the same rule that was measured. When that policy is per-order expected
    value there is no single number to publish — the effective cutoff depends on
    the order's freight — so the serving code re-derives it and the figure here
    is only the median for orientation.

    The prepay line has to come from the score *distribution*, not from
    arithmetic on the confirm line. A calibrated model on a 6.6% base rate
    barely exceeds 0.31, so a band placed at, say, 0.6 would look reasonable
    and never once fire. Anchoring it at the 99th percentile keeps the heavier
    intervention rare and, crucially, reachable — it is the worst 1% of orders
    by definition, whatever the model's scores happen to span.
    """
    prepay = float(np.quantile(p, 0.99))
    if chosen["threshold"] is None:
        # risk x freight x (1+m) x prevention > friction, solved for risk.
        loss = failure_cost(freight, a).to_numpy() * a.prevention_rate
        # A zero-freight order has nothing to protect, so its break-even is
        # infinite — real, but it would poison the median. Report the median
        # over orders where the figure is finite.
        per_order = np.divide(a.false_positive_friction, loss,
                              out=np.full_like(loss, np.nan, dtype=float),
                              where=loss > 0)
        confirm_at = round(float(np.nanmedian(per_order)), 4)
        basis = (
            "confirm = per-order expected value (a different cutoff for every "
            f"order; the median is {confirm_at}); "
        )
    else:
        confirm_at = chosen["threshold"]
        basis = f"confirm = {confirm_at}, chosen on the validation window; "
    return {
        "policy": chosen["policy"],
        "confirm_at": confirm_at,
        "prepay_at": round(max(prepay, confirm_at), 4),
        "basis": basis
        + "prepay = 99th percentile of held-out predicted risk (worst 1% of orders)",
        "max_predicted_risk": round(float(p.max()), 4),
    }


def _policies(y_test, p_test, freight, a, selection: dict) -> list[dict]:
    """Every way of turning a score into a decision, measured side by side.

    Ordered best-first on net saving, so `policies[0]` is what ships. Which one
    wins is a genuine result rather than a foregone conclusion, and the loser is
    reported next to the winner because the reason it loses is the most useful
    thing this project found.
    """
    rows = []

    # 1. One global probability cutoff, tuned on validation. The textbook move.
    r = evaluate_threshold(y_test, p_test, freight, selection["threshold"], a)
    rows.append({**r, "policy": "tuned global threshold",
                 "detail": f"flag when risk >= {selection['threshold']}, "
                           "chosen on the validation window"})

    # 2. A fixed share of orders, that share chosen on validation. Distribution-
    #    relative, so a shift in the level of scores does not move the workload.
    q = 1.0 - selection["validation_flag_rate"]
    cutoff = float(np.quantile(p_test, q))
    r = evaluate_policy(y_test, p_test >= cutoff, freight, a)
    rows.append({**r, "threshold": round(cutoff, 4), "policy": "fixed flag rate",
                 "detail": f"flag the riskiest {selection['validation_flag_rate']:.1%} of "
                           "orders, that share chosen on the validation window"})

    # 3. Per-order expected value. Nothing is tuned at all.
    r = evaluate_policy(y_test, expected_value_flags(p_test, freight, a), freight, a)
    rows.append({**r, "threshold": None, "policy": "per-order expected value",
                 "detail": "flag when risk x freight x prevention > friction, "
                           "which is a different cutoff for every order"})

    return sorted(rows, key=lambda d: -d["net_saving"])


def _score_baselines(test: pd.DataFrame, y_test, freight, a) -> list[dict]:
    """Measure each rule on the same test set, at its own natural threshold."""
    rows = []
    for name, fn in BASELINES.items():
        s = fn(test)
        r = evaluate_threshold(y_test, s, freight, 0.5, a)
        r["name"] = name
        rows.append(r)
    return rows


def main() -> dict:
    t0 = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # Note: `features.add_history` (seller / route / category track records) is
    # deliberately not called. It was built and measured, and degraded held-out
    # performance — see its docstring for the numbers.
    df = build_features(build_order_table())

    train, test, boundary = time_split(df, TEST_FRACTION)

    X_train, y_train = feature_matrix(train), train.target.to_numpy()
    X_test, y_test = feature_matrix(test), test.target.to_numpy()

    # Calibrate across expanding time folds rather than a single held-back
    # slice. A fixed slice would cost 15% of the training data — measurably
    # worse ranking — while TimeSeriesSplit trains each fold on its past and
    # calibrates on its future, so every order contributes and no fold ever
    # sees data from ahead of itself.
    #
    # Platt (sigmoid) rather than isotonic, deliberately. Isotonic is a step
    # function: it collapses distinct scores onto shared values, and those ties
    # flatten PR-AUC — an artefact of the calibrator, not the model. Sigmoid is
    # strictly monotonic, so ranking survives calibration untouched.
    model = CalibratedClassifierCV(
        build_pipeline(), method="sigmoid", cv=TimeSeriesSplit(n_splits=4)
    ).fit(X_train, y_train)

    p_test = model.predict_proba(X_test)[:, 1]
    freight = test["freight_total"]
    a = CostAssumptions()

    # Three ways to turn scores into decisions, all measured the same way and
    # none of them allowed to see the test set first. See `_policies`.
    selection = _choose_threshold(train, X_train, y_train, a)
    policies = _policies(y_test, p_test, freight, a, selection)
    chosen = policies[0]
    # The mask the winning policy actually produces, so every downstream
    # diagnostic describes what ships rather than a threshold nobody uses.
    shipped_flags = (
        expected_value_flags(p_test, freight, a)
        if chosen["threshold"] is None
        else p_test >= chosen["threshold"]
    )

    # The test-set optimum, for reference only. It is what the threshold would
    # have been with hindsight, and the gap between it and `chosen` is the
    # honest cost of having to decide in advance.
    curve, hindsight = cost_curve(y_test, p_test, freight, a)
    at_half = evaluate_threshold(y_test, p_test, freight, 0.5, a)
    baselines = _score_baselines(test, y_test, freight, a)

    metrics = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "source": "Olist Brazilian E-Commerce (Kaggle), ~100k real orders 2016-2018",
            "orders_total": int(len(df)),
            "target": "order not delivered as promised (cancelled, never delivered, or late)",
            "target_rate_overall": round(float(df.target.mean()), 4),
        },
        "split": {
            "method": "chronological; last 20% held out and never used for fitting or tuning",
            "boundary": str(boundary),
            "n_train": int(len(X_train)),
            "calibration": "Platt, across 4 expanding time folds (TimeSeriesSplit)",
            "n_test": int(len(X_test)),
            "target_rate_train": round(float(train.target.mean()), 4),
            "target_rate_test": round(float(test.target.mean()), 4),
        },
        "ranking": {
            "pr_auc": round(float(average_precision_score(y_test, p_test)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, p_test)), 4),
            "brier": round(float(brier_score_loss(y_test, p_test)), 4),
            "positive_rate_test": round(float(y_test.mean()), 4),
        },
        "at_threshold_0.5": at_half,
        "operating_point": chosen,
        "policies": policies,
        "threshold_selection": selection,
        "hindsight_best": hindsight,
        "action_bands": _action_bands(p_test, freight, a, chosen),
        "cost_assumptions": a.describe(),
        "cost_curve": curve,
        "baselines": baselines,
        "n_features": len(ALL_FEATURES),
        # How far the two assumed cost inputs can be wrong before the
        # conclusion flips, and how precise the headline figures actually are.
        "sensitivity": sensitivity_report(y_test, p_test, freight, a),
        "uncertainty": bootstrap_report(test, y_test, p_test, freight, a),
        "diagnostics": full_report(
            model, test, X_test, y_test, p_test, freight, shipped_flags, a
        ),
        "train_seconds": round(time.perf_counter() - t0, 1),
    }

    joblib.dump(model, ARTIFACTS / "model.joblib")
    # A typical order, from the training window only. `actions.explain` scores
    # an order once per feature with that feature swapped to this baseline; the
    # change in risk is the feature's actual contribution for that order.
    _typical_row(X_train).to_json(ARTIFACTS / "baseline_row.json", orient="records")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # Keep the scored test set so the evaluation can be reproduced without
    # retraining, and so the API can serve real example orders.
    test.assign(predicted_risk=p_test).to_parquet(PROCESSED / "test_scored.parquet")

    _print(metrics)
    return metrics


def _print(m: dict) -> None:
    r, b, h = m["ranking"], m["baselines"], m["at_threshold_0.5"]
    best, sel, hind = m["operating_point"], m["threshold_selection"], m["hindsight_best"]
    print("=" * 72)
    print("  PRE-DISPATCH ORDER-FAILURE SCORER")
    print("=" * 72)
    s = m["split"]
    print(f"  train {s['n_train']:,} -> test {s['n_test']:,}   (split at {s['boundary'][:10]})")
    print(f"  failure rate: train {s['target_rate_train']:.1%}  test {s['target_rate_test']:.1%}")
    print()
    print(f"  PR-AUC {r['pr_auc']:.4f}   ROC-AUC {r['roc_auc']:.4f}   Brier {r['brier']:.4f}")
    print(f"  (a random model would score PR-AUC {r['positive_rate_test']:.4f})")
    print()
    print(f"  {'':30}{'prec':>8}{'recall':>9}{'flagged':>10}{'net R$':>12}")
    print(f"  {'model @ 0.5':30}{h['precision']:>8.3f}{h['recall']:>9.3f}"
          f"{h['flag_rate']:>10.1%}{h['net_saving']:>12,.0f}")
    for row in m["policies"]:
        mark = "> " if row is m["policies"][0] else "  "
        print(f"  {mark + row['policy']:30}{row['precision']:>8.3f}"
              f"{row['recall']:>9.3f}{row['flag_rate']:>10.1%}{row['net_saving']:>12,.0f}")
    print(f"  {'  (best in hindsight)':30}{hind['precision']:>8.3f}"
          f"{hind['recall']:>9.3f}{hind['flag_rate']:>10.1%}{hind['net_saving']:>12,.0f}")
    print("  " + "-" * 68)
    for row in b:
        print(f"  {'rule: ' + row['name']:30}{row['precision']:>8.3f}{row['recall']:>9.3f}"
              f"{row['flag_rate']:>10.1%}{row['net_saving']:>12,.0f}")
    print()
    print("  same flag rate, model vs rule (isolates ranking quality):")
    print(f"  {'':30}{'rule P':>9}{'model P':>10}{'lift':>7}")
    for row in m["diagnostics"]["matched_flag_rate"]:
        lift = f"{row['precision_lift']}x" if row["precision_lift"] else "-"
        print(f"  {row['baseline'][:28]:30}{row['rule_precision']:>9.3f}"
              f"{row['model_precision']:>10.3f}{lift:>7}")
    print()
    print(f"  shipping policy               : {best['policy']}")
    print(f"  base rate drift               : {sel['validation_target_rate']:.1%} in the "
          f"validation window -> {m['split']['target_rate_test']:.1%} in test")
    gap = hind["net_saving"] - best["net_saving"]
    print("  vs best single threshold      : R$ "
          + (f"{gap:,.0f} left on the table by deciding in advance" if gap > 0
             else f"{-gap:,.0f} better than the best threshold hindsight could pick"))
    print(f"  false-positive cost           : R$ {best['false_positive_cost']:,.0f}"
          f"   ({best['fp']:,} good orders flagged)")
    print(f"  net saving per 1,000 orders   : R$ {best['net_saving_per_1k_orders']:,.0f}")
    sens, unc = m["sensitivity"], m["uncertainty"]
    hd, mult = sens["friction_headroom"], sens["friction_headroom_multiple"]
    print(f"  friction headroom             : profitable up to R$ {hd}"
          f"  ({mult}x the assumed R$ {sens['shipped_assumptions']['friction']:.0f})"
          if hd else "  friction headroom             : never profitable")
    pr, rc = unc["precision"], unc["recall"]
    print(f"  95% CI  precision {best['precision']:.3f} [{pr['lo']:.3f}, {pr['hi']:.3f}]"
          f"   recall {best['recall']:.3f} [{rc['lo']:.3f}, {rc['hi']:.3f}]")
    for name, d in unc["matched_flag_rate_precision_diff"].items():
        if name == "flag_all":
            continue
        print(f"    vs {name[:26]:28}{d['point']:+.3f} "
              f"[{d['lo']:+.3f}, {d['hi']:+.3f}]  {d['verdict']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
