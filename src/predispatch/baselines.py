"""Simple rules the model has to beat.

A model is only worth deploying if it outperforms what a merchant could do with
a spreadsheet and an afternoon. Skipping this step is how projects end up
reporting an impressive-looking AUC for something a single `if` statement
matches.

Each baseline returns a score in [0, 1] so it can be measured on exactly the
same footing as the model — same test set, same threshold sweep, same cost
curve. If the model only edges past these, that is the finding, and it gets
reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def flag_all(df: pd.DataFrame) -> np.ndarray:
    """Flag everything. The degenerate ceiling: perfect recall, useless precision.

    Worth measuring because it puts a floor under what "catching failures" means
    — any model must beat blanket suspicion on *cost*, not just on recall.
    """
    return np.ones(len(df), dtype=float)


def cross_state(df: pd.DataFrame) -> np.ndarray:
    """Flag orders shipping outside the seller's state.

    The obvious human heuristic: long hauls go wrong more often.
    """
    return (df["same_state"] == 0).astype(float).to_numpy()


def long_distance(df: pd.DataFrame, quantile: float = 0.75) -> np.ndarray:
    """Flag the longest-distance quarter of orders."""
    d = df["distance_km"].fillna(df["distance_km"].median())
    return (d >= d.quantile(quantile)).astype(float).to_numpy()


def tight_promise(df: pd.DataFrame, quantile: float = 0.25) -> np.ndarray:
    """Flag orders with the shortest promised delivery windows.

    A short promise is the easiest one to break, so this is the sharpest rule a
    merchant could write without any modelling.
    """
    d = df["promised_days"].fillna(df["promised_days"].median())
    return (d <= d.quantile(quantile)).astype(float).to_numpy()


BASELINES = {
    "flag_all": flag_all,
    "cross_state": cross_state,
    "long_distance_top25pct": long_distance,
    "tight_promise_bottom25pct": tight_promise,
}
