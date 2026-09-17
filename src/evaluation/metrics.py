"""Shared evaluation metrics for demand forecasting and ETA prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def medae(y_true, y_pred) -> float:
    return float(np.median(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def p90_abs_error(y_true, y_pred) -> float:
    return float(np.percentile(np.abs(np.asarray(y_true) - np.asarray(y_pred)), 90))


def p95_abs_error(y_true, y_pred) -> float:
    return float(np.percentile(np.abs(np.asarray(y_true) - np.asarray(y_pred)), 95))


def wape(y_true, y_pred) -> float:
    """Weighted Absolute Percentage Error -- robust to zero-demand buckets,
    unlike MAPE which blows up when y_true == 0 (common for low-volume
    restaurants/zones in this dataset)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def regression_report(y_true, y_pred) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MedAE": medae(y_true, y_pred),
        "P90_AbsError": p90_abs_error(y_true, y_pred),
        "P95_AbsError": p95_abs_error(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
    }


def interval_coverage(y_true, lower, upper) -> float:
    """Fraction of true values that fall within [lower, upper] -- used to
    validate quantile-regression / conformal prediction intervals."""
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def mean_interval_width(lower, upper) -> float:
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def time_based_split(
    df: pd.DataFrame, time_col: str, train_frac: float, val_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered dataframe into train/val/test by TIME, never by
    random shuffling -- shuffling a temporal forecasting dataset lets the
    model see future information during training via nearby lag/rolling
    features that reference adjacent rows, which silently leaks. Splitting on
    absolute time boundaries (earliest 70% / next 15% / latest 15%) guarantees
    the test set is later than everything the model was fit on.
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def error_breakdown(df: pd.DataFrame, y_true_col: str, y_pred_col: str, group_col: str) -> pd.DataFrame:
    """Per-group (e.g. zone, peak/off-peak, distance bucket) error breakdown --
    a single overall metric hides whether the model is systematically worse
    for a business-critical segment (e.g. long-distance or peak-hour orders)."""
    rows = []
    for key, g in df.groupby(group_col):
        rows.append(
            {
                group_col: key,
                "n": len(g),
                **regression_report(g[y_true_col], g[y_pred_col]),
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
