"""
Tests for app/inference.py's input handling and error behavior on edge cases.
Uses the small toy panels rather than the full generated dataset so these
run fast and don't depend on data/ being pre-generated.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.inference import predict_demand, predict_eta
from src.utils.config import load_config


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    config = load_config()
    config["paths"]["processed_dir"] = str(tmp_path)
    return config


def _write_toy_demand_panel(path):
    bucket_ts = pd.date_range("2026-01-01", periods=200, freq="30min")
    df = pd.DataFrame({
        "restaurant_id": 0,
        "zone_id": 0,
        "bucket_ts": bucket_ts,
        "order_count": [i % 10 for i in range(200)],
        "avg_basket_value": 300.0,
        "avg_distance_km": 3.0,
        "hour": bucket_ts.hour,
        "dow": bucket_ts.dayofweek,
        "is_weekend": (bucket_ts.dayofweek >= 5).astype(int),
        "month": bucket_ts.month,
        "day_of_year": bucket_ts.dayofyear,
        "is_lunch_peak": bucket_ts.hour.isin([12, 13, 14]).astype(int),
        "is_dinner_peak": bucket_ts.hour.isin([19, 20, 21]).astype(int),
        "hour_sin": 0.0, "hour_cos": 0.0, "dow_sin": 0.0, "dow_cos": 0.0,
        "weather_condition": "Clear",
        "traffic_multiplier": 1.0,
        "rain_mm": 0.0,
    })
    for lag_col in ["lag_30m", "lag_1h", "lag_2h", "lag_1d", "lag_7d"]:
        df[lag_col] = df["order_count"].shift(1).fillna(0)
    for roll_col in ["roll_mean_2h", "roll_std_2h", "roll_max_2h",
                     "roll_mean_8h", "roll_std_8h", "roll_max_8h",
                     "roll_mean_1d", "roll_std_1d", "roll_max_1d", "ewm_mean_short"]:
        df[roll_col] = df["order_count"].shift(1).fillna(0)
    df["restaurant_hist_avg_demand"] = df["order_count"].shift(1).expanding().mean().fillna(0)
    df["restaurant_hist_volatility"] = 1.0
    df["zone_demand_prev_bucket"] = df["order_count"].shift(1).fillna(0)
    for target_col, shift in [("target_30m", -1), ("target_1h", -2), ("target_2h", -4)]:
        df[target_col] = df["order_count"].shift(shift)
    df.to_parquet(path)


def test_predict_demand_raises_clear_error_for_unknown_restaurant(cfg, tmp_path):
    _write_toy_demand_panel(tmp_path / "demand_timeseries.parquet")
    with pytest.raises(ValueError, match="No feature row found"):
        predict_demand(restaurant_id=999, timestamp="2026-01-05 12:00", cfg=cfg)


def test_predict_demand_raises_clear_error_for_out_of_range_timestamp(cfg, tmp_path):
    _write_toy_demand_panel(tmp_path / "demand_timeseries.parquet")
    with pytest.raises(ValueError, match="No feature row found"):
        predict_demand(restaurant_id=0, timestamp="2030-01-01 12:00", cfg=cfg)


def test_predict_demand_returns_expected_schema(cfg, tmp_path):
    _write_toy_demand_panel(tmp_path / "demand_timeseries.parquet")
    result = predict_demand(restaurant_id=0, timestamp="2026-01-03 12:00", cfg=cfg)
    for key in ["forecast_30m", "forecast_1h", "forecast_2h", "expected_peak"]:
        assert key in result
    assert isinstance(result["expected_peak"], bool)
    assert result["forecast_30m"] >= 0


def test_predict_eta_raises_for_unknown_restaurant(cfg, tmp_path):
    ts = pd.date_range("2026-01-01", periods=50, freq="h")
    eta_df = pd.DataFrame({
        "order_timestamp": ts,
        "restaurant_id": 0,
        "zone_id": 0,
        "cuisine": "North Indian",
        "popularity_score": 1.0,
        "base_prep_time_min": 15.0,
        "rating": 4.2,
        "distance_km": 3.0,
        "n_items": 2,
        "basket_value_inr": 300.0,
        "eta_minutes": 25.0,
        "traffic_multiplier": 1.0,
        "weather_condition": "Clear",
        "hour": ts.hour,
        "dow": ts.dayofweek,
        "is_weekend": (ts.dayofweek >= 5).astype(int),
        "is_peak_hour": ts.hour.isin([12, 13, 14, 19, 20, 21]).astype(int),
        "hour_sin": 0.0,
        "hour_cos": 0.0,
        "restaurant_hist_avg_prep_time": 15.0,
        "restaurant_hist_avg_eta": 25.0,
        "restaurant_order_sequence_num": range(50),
    })
    eta_df.to_parquet(tmp_path / "eta_features.parquet")
    with pytest.raises(ValueError, match="No history found"):
        predict_eta(
            restaurant_id=42, distance_km=3.0, n_items=2, basket_value_inr=300.0,
            timestamp="2026-01-10 12:00", cfg=cfg,
        )
