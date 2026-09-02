"""Tests for the claims this project makes.

Each one guards a specific way the headline numbers could be wrong, rather than
exercising code for its own sake. The leakage and time-split tests matter most:
both failure modes make the metrics look *better*, so nothing about a broken run
would appear broken.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from predispatch.config import ARTIFACTS, LEAKY_COLUMNS, TEST_FRACTION
from predispatch.actions import PREPAY, SHIP, decide
from predispatch.cost import (
    CostAssumptions,
    cost_curve,
    evaluate_threshold,
    expected_value_flags,
)
from predispatch.data import build_order_table, time_split
from predispatch.features import (
    ALL_FEATURES,
    build_features,
    feature_matrix,
    leakage_audit,
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return build_features(build_order_table())


# --- leakage ---------------------------------------------------------------

def test_no_post_outcome_column_is_a_feature():
    """The feature list itself must not name anything recorded after dispatch."""
    assert not (set(ALL_FEATURES) & set(LEAKY_COLUMNS))


def test_leakage_audit_rejects_a_leaked_column():
    """The guard must actually fire — a check that never fails protects nothing."""
    with pytest.raises(ValueError, match="Post-outcome"):
        leakage_audit(list(ALL_FEATURES) + ["order_delivered_customer_date"])


def test_feature_matrix_excludes_outcomes(frame):
    """The joined frame legitimately carries the label; the matrix must not."""
    assert {"target", "order_status"} <= set(frame.columns)
    assert not (set(feature_matrix(frame).columns) & set(LEAKY_COLUMNS))


# --- the split -------------------------------------------------------------

def test_time_split_does_not_overlap(frame):
    """Every test order must be placed at or after every training order."""
    train, test, boundary = time_split(frame, TEST_FRACTION)
    assert train.order_purchase_timestamp.max() <= test.order_purchase_timestamp.min()
    assert boundary == train.order_purchase_timestamp.max()
    assert len(train) + len(test) == len(frame)


def test_split_is_not_random(frame):
    """A random split would leave both sides spanning the same period.

    This is the difference between a model judged on its future and one judged
    on a reshuffle of its own training window.
    """
    train, test, _ = time_split(frame, TEST_FRACTION)
    assert test.order_purchase_timestamp.min() > train.order_purchase_timestamp.min()


# --- the target ------------------------------------------------------------

def test_target_is_the_union_of_its_three_parts(frame):
    parts = (
        frame.cancelled_or_unavailable | frame.never_delivered | frame.delivered_late
    )
    assert (frame.target == parts.astype(int)).all()


def test_target_rate_is_plausible(frame):
    """Guards against a join or filter silently changing the problem."""
    assert 0.05 < frame.target.mean() < 0.20


# --- cost model ------------------------------------------------------------

def test_flagging_nothing_saves_nothing():
    y = np.array([1, 0, 1, 0])
    p = np.zeros(4)
    freight = pd.Series([10.0, 10.0, 10.0, 10.0])
    r = evaluate_threshold(y, p, freight, 1.1, CostAssumptions())
    assert r["flagged"] == 0
    assert r["net_saving"] == 0.0


def test_flagging_everything_costs_friction():
    """Perfect recall is not free — blanket suspicion pays friction on every order."""
    y = np.array([1, 0, 0, 0])
    p = np.ones(4)
    freight = pd.Series([1.0, 1.0, 1.0, 1.0])
    a = CostAssumptions(false_positive_friction=50.0)
    r = evaluate_threshold(y, p, freight, 0.5, a)
    assert r["recall"] == 1.0
    assert r["net_saving"] < 0


def test_cost_optimum_is_at_least_as_good_as_any_swept_point():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.1, 2000)
    p = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 2000), 0, 1)
    freight = pd.Series(rng.uniform(5, 40, 2000))
    curve, best = cost_curve(y, p, freight)
    assert best["net_saving"] == max(r["net_saving"] for r in curve)


# --- reported numbers ------------------------------------------------------

@pytest.mark.skipif(
    not (ARTIFACTS / "metrics.json").exists(), reason="run python -m predispatch.train"
)
def test_reported_metrics_are_internally_consistent():
    """Catches a stale metrics file left behind by an older run."""
    m = json.loads((ARTIFACTS / "metrics.json").read_text())
    best, ranking = m["operating_point"], m["ranking"]

    # The stored rate is rounded to four places, so allow a rounding order.
    assert abs(
        best["tp"] + best["fn"] - ranking["positive_rate_test"] * m["split"]["n_test"]
    ) <= 1
    assert best["net_saving"] >= m["at_threshold_0.5"]["net_saving"]
    # A model worth deploying must beat chance on the metric that suits a rare
    # positive class.
    assert ranking["pr_auc"] > ranking["positive_rate_test"]
    assert m["diagnostics"]["pr_auc_check"] == ranking["pr_auc"]


@pytest.mark.skipif(
    not (ARTIFACTS / "metrics.json").exists(), reason="run python -m predispatch.train"
)
def test_model_beats_every_baseline_on_cost():
    """The claim the project rests on, asserted rather than assumed."""
    m = json.loads((ARTIFACTS / "metrics.json").read_text())
    best = m["operating_point"]["net_saving"]
    for row in m["baselines"]:
        assert best > row["net_saving"], f"baseline {row['name']} matched the model"


# --- structural leakage ----------------------------------------------------

def test_no_empty_basket_orders_survive(frame):
    """Orders with no `order_items` row must be gone before modelling.

    They are 100% failures, because the items table is only written once an
    order is picked — so an empty basket is an outcome, not an input. This
    single filter is what separates an honest precision from a fake 1.000.
    """
    assert frame.price_total.notna().all()
    assert frame.freight_total.notna().all()


def test_no_test_order_is_perfectly_separable(frame):
    """A feature that alone identifies failures is a leak, not a signal.

    Guards the general case rather than the one instance already fixed: if any
    numeric feature's null-ness predicts the target perfectly, something
    post-outcome has got in again.
    """
    for col in ALL_FEATURES:
        nulls = frame[col].isna()
        if 20 < nulls.sum() < len(frame) - 20:
            assert frame.loc[nulls, "target"].mean() < 0.99, (
                f"null {col} implies failure — check for structural leakage"
            )


# --- threshold selection ---------------------------------------------------

@pytest.mark.skipif(
    not (ARTIFACTS / "metrics.json").exists(), reason="run python -m predispatch.train"
)
def test_no_threshold_policy_was_tuned_on_the_test_set():
    """Any global threshold we ship must come from validation, not hindsight.

    `hindsight_best` sweeps the test set itself, so no threshold chosen without
    it can beat it. If one does, the selection code is reading test labels.

    The per-order policy is deliberately exempt: it has no threshold to tune, so
    there is nothing for hindsight to have an advantage over — and on this data
    it does in fact beat the hindsight optimum, which is the point.
    """
    m = json.loads((ARTIFACTS / "metrics.json").read_text())
    hindsight, sel = m["hindsight_best"], m["threshold_selection"]

    tuned = next(r for r in m["policies"] if r["policy"] == "tuned global threshold")
    assert tuned["threshold"] == sel["threshold"]

    for row in m["policies"]:
        if row["threshold"] is not None:
            assert hindsight["net_saving"] >= row["net_saving"], (
                f"{row['policy']} beat the test set's own optimum — it must have "
                "seen the test labels"
            )


@pytest.mark.skipif(
    not (ARTIFACTS / "metrics.json").exists(), reason="run python -m predispatch.train"
)
def test_every_action_band_is_reachable():
    """A band above the model's highest score would never once fire.

    Easy to ship without noticing: the thresholds look sensible in isolation
    and the endpoint returns valid answers, it just silently never recommends
    the strongest action.
    """
    m = json.loads((ARTIFACTS / "metrics.json").read_text())
    bands = m["action_bands"]
    assert bands["confirm_at"] <= bands["prepay_at"] <= bands["max_predicted_risk"]


# --- the decision layer ----------------------------------------------------

def test_the_shipped_policy_is_the_one_that_was_measured():
    """Serving must apply the rule the reported metrics describe.

    The failure this guards is silent and severe: the API keeps returning
    plausible answers while the published precision and recall no longer
    describe anything it does.
    """
    m = json.loads((ARTIFACTS / "metrics.json").read_text())
    assert m["action_bands"]["policy"] == m["operating_point"]["policy"]
    assert m["operating_point"] is not None


def test_expected_value_rule_is_freight_sensitive():
    """The same risk must decide differently on different freight.

    This is the entire claim behind the per-order policy, so it is worth an
    assertion rather than a comment: a heavy parcel is worth protecting at a
    risk that would not justify touching a cheap one.
    """
    a = CostAssumptions()
    p = np.array([0.10, 0.10])
    freight = pd.Series([100.0, 0.5])
    flags = expected_value_flags(p, freight, a)
    assert flags[0] and not flags[1]


def test_low_risk_cheap_order_ships():
    policy = {
        "policy": "per-order expected value", "confirm_at": 0.2, "prepay_at": 0.9,
        "friction": 5.0, "prevention_rate": 0.7, "return_multiplier": 1.0,
    }
    assert decide(0.01, 1.0, policy) is SHIP


def test_high_risk_expensive_order_escalates_to_prepay():
    policy = {
        "policy": "per-order expected value", "confirm_at": 0.2, "prepay_at": 0.25,
        "friction": 5.0, "prevention_rate": 0.7, "return_multiplier": 1.0,
    }
    assert decide(0.40, 80.0, policy) is PREPAY


# --- robustness of the reported claims -------------------------------------

@pytest.mark.skipif(
    not (ARTIFACTS / "metrics.json").exists(), reason="run python -m predispatch.train"
)
def test_friction_headroom_is_the_first_crossing_not_a_later_flicker():
    """Headroom must be a friction at which the policy is genuinely profitable.

    The first implementation bisected, which assumes net saving falls
    monotonically with friction. It does not: as friction rises the flag rate
    collapses, and the saving then rattles around zero on the handful of orders
    left. Bisection latched onto a late positive flicker and reported 11x the
    real headroom. This asserts the reported figure is profitable and that the
    next step up is not.
    """
    import pandas as pd
    from predispatch.config import PROCESSED
    from predispatch.sensitivity import _at

    m = json.loads((ARTIFACTS / "metrics.json").read_text())
    hd = m["sensitivity"]["friction_headroom"]
    if hd is None:
        pytest.skip("policy never profitable")

    d = pd.read_parquet(PROCESSED / "test_scored.parquet")
    y, p, fr = d.target.to_numpy(), d.predicted_risk.to_numpy(), d.freight_total
    a = CostAssumptions()

    assert _at(y, p, fr, hd, a.prevention_rate, a)["net_saving"] > 0
    assert _at(y, p, fr, hd + 0.25, a.prevention_rate, a)["net_saving"] <= 0


@pytest.mark.skipif(
    not (ARTIFACTS / "metrics.json").exists(), reason="run python -m predispatch.train"
)
def test_confidence_intervals_bracket_their_point_estimates():
    """An interval that does not contain the reported figure is a bug, not a result."""
    m = json.loads((ARTIFACTS / "metrics.json").read_text())
    op, u = m["operating_point"], m["uncertainty"]
    for key, point in [("precision", op["precision"]),
                       ("recall", op["tp"] / (op["tp"] + op["fn"])),
                       ("pr_auc", m["ranking"]["pr_auc"]),
                       ("flag_rate", op["flag_rate"])]:
        ci = u[key]
        assert ci["lo"] <= point <= ci["hi"], f"{key}: {point} outside [{ci['lo']}, {ci['hi']}]"


@pytest.mark.skipif(
    not (ARTIFACTS / "metrics.json").exists(), reason="run python -m predispatch.train"
)
def test_ranking_beats_chance_with_the_interval_not_just_the_point():
    """PR-AUC must clear the base rate at the lower bound, not only on average."""
    m = json.loads((ARTIFACTS / "metrics.json").read_text())
    assert m["uncertainty"]["pr_auc"]["lo"] > m["ranking"]["positive_rate_test"]


def test_sensitivity_reprices_decisions_rather_than_just_outcomes():
    """Changing friction must change *what gets flagged*, not only its cost.

    A sweep that held the flags fixed would measure a policy nobody would run —
    a merchant who believed friction were higher would also flag less.
    """
    import pandas as pd
    from predispatch.sensitivity import _at

    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.1, 3000)
    p = np.clip(y * 0.25 + rng.normal(0.15, 0.1, 3000), 0.001, 0.999)
    freight = pd.Series(rng.uniform(5, 50, 3000))
    a = CostAssumptions()

    cheap = _at(y, p, freight, 1.0, 0.7, a)
    dear = _at(y, p, freight, 25.0, 0.7, a)
    assert cheap["flag_rate"] > dear["flag_rate"]


# --- rolling backtest ------------------------------------------------------

BACKTEST = ARTIFACTS / "backtest.json"


@pytest.mark.skipif(not BACKTEST.exists(), reason="run python -m predispatch.backtest")
def test_backtest_windows_are_disjoint():
    """Test windows must not share orders, or five results are not five results.

    The first version sized the window wider than the stride between cuts, so
    consecutive windows overlapped by roughly 1,200 orders. Not leakage — each
    window still trained only on its own past — but it quietly turned five
    independent observations into five correlated ones, which is exactly the
    property the backtest exists to provide.
    """
    b = json.loads(BACKTEST.read_text())
    w = b["windows"]
    assert len(w) == b["summary"]["n_windows"]

    for prev, cur in zip(w, w[1:]):
        assert cur["train_n"] > prev["train_n"], "training window must expand"
        assert cur["test_from"] > prev["test_to"], (
            f"window {cur['window']} starts {cur['test_from']} but window "
            f"{prev['window']} runs to {prev['test_to']} — they overlap"
        )


@pytest.mark.skipif(not BACKTEST.exists(), reason="run python -m predispatch.backtest")
def test_backtest_each_window_beats_chance_on_ranking():
    """Ranking should hold up everywhere, even where the money does not.

    Separating these two matters: PR-AUC above the base rate says the model
    orders orders usefully, which is a weaker and more durable claim than the
    net saving being positive in any given window.
    """
    b = json.loads(BACKTEST.read_text())
    for x in b["windows"]:
        assert x["pr_auc"] > x["chance"], (
            f"window {x['window']} ranks no better than chance "
            f"({x['pr_auc']} vs {x['chance']})"
        )
