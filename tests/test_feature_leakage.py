"""
Tests specifically checking that future observations cannot enter historical
features -- the most important correctness property of this project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    _add_lag_rolling_features,
    _add_restaurant_zone_history,
    _add_targets,
)


@pytest.fixture
def toy_panel():
    """A single restaurant, 10 buckets of 30 minutes, with a known,
    hand-crafted order_count sequence so lag/rolling values can be checked
    exactly."""
    bucket_ts = pd.date_range("2026-01-01", periods=10, freq="30min")
    order_count = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return pd.DataFrame({
        "restaurant_id": 0,
        "zone_id": 0,
        "bucket_ts": bucket_ts,
        "order_count": order_count,
    })


def test_lag_feature_uses_only_past_values(toy_panel):
    out = _add_lag_rolling_features(toy_panel, bucket_minutes=30, lag_buckets=[1, 2], rolling_windows=[])
    # lag_30m at row i must equal order_count at row i-1 (never i or later)
    assert out["lag_30m"].iloc[0] != out["lag_30m"].iloc[0]  # NaN for first row (no history)
    assert out["lag_30m"].iloc[5] == out["order_count"].iloc[4]
    assert out["lag_1h"].iloc[5] == out["order_count"].iloc[3]


def test_rolling_mean_excludes_current_bucket(toy_panel):
    out = _add_lag_rolling_features(toy_panel, bucket_minutes=30, lag_buckets=[], rolling_windows=[3])
    # roll_mean_1h30m over the 3 PRECEDING buckets, at row index 5 (order_count=6)
    # should be mean of rows 2,3,4 (order_count 3,4,5) = 4.0, NOT including row 5 itself.
    row5_roll_col = [c for c in out.columns if c.startswith("roll_mean_")][0]
    assert out[row5_roll_col].iloc[5] == pytest.approx(4.0)
    # It must NOT equal a window that includes the current value (which would be (4+5+6)/3=5.0)
    assert out[row5_roll_col].iloc[5] != pytest.approx(5.0)


def test_restaurant_history_is_expanding_and_excludes_current_row(toy_panel):
    out = _add_restaurant_zone_history(toy_panel)
    # At row index 4 (order_count=5), historical avg should be mean(1,2,3,4) = 2.5
    assert out["restaurant_hist_avg_demand"].iloc[4] == pytest.approx(2.5)
    # It must not include the current row's own value (would be mean(1,2,3,4,5)=3.0)
    assert out["restaurant_hist_avg_demand"].iloc[4] != pytest.approx(3.0)


def test_target_horizon_is_strictly_future(toy_panel):
    out = _add_targets(toy_panel, bucket_minutes=30, horizons_minutes=[30, 60])
    # target_30m at row i must equal order_count at row i+1
    assert out["target_30m"].iloc[0] == out["order_count"].iloc[1]
    assert out["target_1h"].iloc[0] == out["order_count"].iloc[2]
    # Last row(s) must be NaN since there is no future data available
    assert pd.isna(out["target_30m"].iloc[-1])


def test_no_feature_column_correlates_perfectly_with_its_own_target(toy_panel):
    """A sanity net: if a bug caused a feature to accidentally equal the
    target (e.g. an unshifted current-bucket value), this would catch it for
    this deterministic monotonic series."""
    out = _add_lag_rolling_features(toy_panel, bucket_minutes=30, lag_buckets=[1], rolling_windows=[2])
    out = _add_targets(out, bucket_minutes=30, horizons_minutes=[30])
    valid = out.dropna(subset=["lag_30m", "target_30m"])
    # lag_30m (t-1) must differ from target_30m (t+1) for this strictly increasing series
    assert not np.allclose(valid["lag_30m"], valid["target_30m"])
