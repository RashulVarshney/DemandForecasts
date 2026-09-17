"""Tests for evaluation metrics and time-based splitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    mae, rmse, medae, p90_abs_error, wape, interval_coverage,
    mean_interval_width, time_based_split, error_breakdown,
)


def test_mae_rmse_basic():
    y_true = [10, 20, 30]
    y_pred = [12, 18, 33]
    assert mae(y_true, y_pred) == pytest.approx((2 + 2 + 3) / 3)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))


def test_wape_handles_all_zero_actuals():
    # If all true values are 0, WAPE's denominator is 0 -- must return NaN,
    # not raise or silently return 0/0=inf.
    result = wape([0, 0, 0], [1, 2, 3])
    assert np.isnan(result)


def test_wape_normal_case():
    y_true = [10, 0, 10]
    y_pred = [12, 1, 8]
    # sum(|error|) = 2 + 1 + 2 = 5, sum(|actual|) = 20
    assert wape(y_true, y_pred) == pytest.approx(5 / 20)


def test_p90_abs_error_matches_percentile():
    y_true = np.arange(100)
    y_pred = y_true + 1  # constant error of 1 except...
    y_pred[-1] += 100  # one huge outlier
    result = p90_abs_error(y_true, y_pred)
    assert result >= 1  # dominated by the bulk, not the single outlier at p90


def test_interval_coverage_and_width():
    y_true = [5, 10, 15]
    lower = [4, 9, 20]   # last one does NOT cover
    upper = [6, 11, 25]
    assert interval_coverage(y_true, lower, upper) == pytest.approx(2 / 3)
    assert mean_interval_width(lower, upper) == pytest.approx(np.mean([2, 2, 5]))


def test_time_based_split_never_shuffles_and_respects_fractions():
    df = pd.DataFrame({
        "t": pd.date_range("2026-01-01", periods=100, freq="h"),
        "value": np.random.RandomState(0).permutation(100),  # scrambled input order
    })
    # Feed it in scrambled row order to ensure the function sorts by time itself.
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)

    train, val, test = time_based_split(df, "t", train_frac=0.7, val_frac=0.15)
    assert len(train) + len(val) + len(test) == len(df)
    # Every timestamp in train must be earlier than every timestamp in val/test.
    assert train["t"].max() < val["t"].min()
    assert val["t"].max() < test["t"].min()
    assert len(train) == 70


def test_error_breakdown_groups_correctly():
    df = pd.DataFrame({
        "y_true": [10, 20, 10, 20],
        "y_pred": [12, 22, 8, 18],
        "segment": ["a", "b", "a", "b"],
    })
    breakdown = error_breakdown(df, "y_true", "y_pred", "segment")
    assert set(breakdown["segment"]) == {"a", "b"}
    assert breakdown.set_index("segment").loc["a", "n"] == 2
    assert breakdown.set_index("segment").loc["a", "MAE"] == pytest.approx(2.0)
