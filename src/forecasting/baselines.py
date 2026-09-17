"""
Baseline demand forecasters. These must be beaten by any "sophisticated" ML
model before that model is worth shipping -- see reports/model_report.md for
the comparison table.

All baselines operate on the demand panel produced by
`src/features/build_features.py` (columns: restaurant_id, bucket_ts,
order_count, lag_30m, lag_1h, lag_1d, lag_7d, ...) and predict a given
target horizon column (e.g. "target_30m").
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def naive_previous_period(df: pd.DataFrame, horizon_lag_col: str) -> np.ndarray:
    """Predict target_h = value observed `h` ago (i.e. repeat the last known
    value). Uses the matching lag column already computed upstream, e.g. for
    a 30-min horizon this is `lag_30m` -- carrying forward "whatever just
    happened will happen again"."""
    return df[horizon_lag_col].values


def same_hour_previous_day(df: pd.DataFrame) -> np.ndarray:
    """Predict using the value from the same bucket exactly 1 day ago."""
    return df["lag_1d"].values


def same_hour_previous_week(df: pd.DataFrame) -> np.ndarray:
    """Predict using the value from the same bucket exactly 1 week ago."""
    return df["lag_7d"].values


def moving_average(df: pd.DataFrame, window_col: str = "roll_mean_2h") -> np.ndarray:
    """Predict using a trailing rolling mean (already computed upstream, using
    only past buckets)."""
    return df[window_col].values


BASELINE_LAG_FOR_HORIZON = {
    "target_30m": "lag_30m",
    "target_1h": "lag_1h",
    "target_2h": "lag_2h",
    "target_6h": "roll_mean_2h",   # no direct 6h lag bucket configured; nearest sensible proxy
    "target_1d": "lag_1d",
}


def run_all_baselines(df: pd.DataFrame, target_col: str) -> dict[str, np.ndarray]:
    lag_col = BASELINE_LAG_FOR_HORIZON.get(target_col, "lag_1h")
    return {
        "naive_previous_period": naive_previous_period(df, lag_col),
        "same_hour_previous_day": same_hour_previous_day(df),
        "same_hour_previous_week": same_hour_previous_week(df),
        "moving_average_2h": moving_average(df, "roll_mean_2h"),
    }
