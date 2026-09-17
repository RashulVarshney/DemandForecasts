"""
Feature engineering for the demand-forecasting panel and the ETA modeling table.

Leakage discipline
-------------------
Every feature is computed using only information strictly available *before*
the timestamp being predicted:
  - Lags and rolling statistics are shifted by at least one full bucket
    (`.shift(1)` before any `.rolling(...)`), so the bucket being predicted
    never contributes to its own features.
  - Zone/restaurant "historical" aggregates (popularity, volatility) are
    computed on an expanding window up to (but excluding) the current bucket.
  - Weather/traffic joined for a given bucket uses only the value observed
    for the current hour (which is realistic: current weather is observable
    in real time, unlike future weather, which would require a forecast
    feed — a caveat documented in reports/model_report.md).

Run:
    python -m src.features.build_features --raw data/raw --out data/processed
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

from src.utils.config import load_config


def _build_demand_panel(orders: pd.DataFrame, bucket_minutes: int) -> pd.DataFrame:
    orders = orders[~orders["is_cancelled"]].copy()
    orders["bucket_ts"] = orders["order_timestamp"].dt.floor(f"{bucket_minutes}min")

    panel = (
        orders.groupby(["restaurant_id", "zone_id", "bucket_ts"])
        .agg(
            order_count=("order_id", "count"),
            avg_basket_value=("basket_value_inr", "mean"),
            avg_distance_km=("distance_km", "mean"),
        )
        .reset_index()
    )

    # Full grid: every (restaurant, bucket) combination, including zero-demand buckets.
    all_buckets = pd.date_range(
        panel["bucket_ts"].min(), panel["bucket_ts"].max(), freq=f"{bucket_minutes}min"
    )
    rest_zone = orders[["restaurant_id", "zone_id"]].drop_duplicates()
    grid = rest_zone.merge(pd.DataFrame({"bucket_ts": all_buckets}), how="cross")

    panel = grid.merge(panel, on=["restaurant_id", "zone_id", "bucket_ts"], how="left")
    panel["order_count"] = panel["order_count"].fillna(0).astype(int)
    panel = panel.sort_values(["restaurant_id", "bucket_ts"]).reset_index(drop=True)
    return panel


def _add_temporal_features(panel: pd.DataFrame) -> pd.DataFrame:
    ts = panel["bucket_ts"]
    panel["hour"] = ts.dt.hour
    panel["dow"] = ts.dt.dayofweek
    panel["is_weekend"] = (panel["dow"] >= 5).astype(int)
    panel["month"] = ts.dt.month
    panel["day_of_year"] = ts.dt.dayofyear
    panel["is_lunch_peak"] = panel["hour"].between(12, 14).astype(int)
    panel["is_dinner_peak"] = panel["hour"].between(19, 21).astype(int)
    # Cyclical encodings so the model sees hour=23 and hour=0 as close.
    panel["hour_sin"] = np.sin(2 * np.pi * panel["hour"] / 24)
    panel["hour_cos"] = np.cos(2 * np.pi * panel["hour"] / 24)
    panel["dow_sin"] = np.sin(2 * np.pi * panel["dow"] / 7)
    panel["dow_cos"] = np.cos(2 * np.pi * panel["dow"] / 7)
    return panel


def _add_lag_rolling_features(
    panel: pd.DataFrame, bucket_minutes: int, lag_buckets: list[int], rolling_windows: list[int]
) -> pd.DataFrame:
    panel = panel.sort_values(["restaurant_id", "bucket_ts"]).reset_index(drop=True)
    g = panel.groupby("restaurant_id")["order_count"]

    for lag in lag_buckets:
        minutes = lag * bucket_minutes
        label = _minutes_to_label(minutes)
        panel[f"lag_{label}"] = g.shift(lag)

    # Rolling stats computed on the ALREADY-SHIFTED series (shift(1) first) so the
    # current bucket's own order_count never leaks into its own rolling features.
    shifted = g.shift(1)
    for window in rolling_windows:
        minutes = window * bucket_minutes
        label = _minutes_to_label(minutes)
        roll = shifted.groupby(panel["restaurant_id"]).rolling(window, min_periods=1)
        panel[f"roll_mean_{label}"] = roll.mean().reset_index(level=0, drop=True)
        panel[f"roll_std_{label}"] = roll.std().reset_index(level=0, drop=True)
        panel[f"roll_max_{label}"] = roll.max().reset_index(level=0, drop=True)

    # Exponentially weighted mean of the shifted series (captures recent momentum).
    panel["ewm_mean_short"] = shifted.groupby(panel["restaurant_id"]).transform(
        lambda s: s.ewm(span=8, min_periods=1).mean()
    )

    return panel


def _minutes_to_label(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes}m"
    if minutes < 1440:
        return f"{minutes // 60}h"
    return f"{minutes // 1440}d"


def _add_restaurant_zone_history(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["restaurant_id", "bucket_ts"]).reset_index(drop=True)

    # Expanding (cumulative) mean of PAST order_count only -> "historical popularity".
    shifted = panel.groupby("restaurant_id")["order_count"].shift(1)
    panel["restaurant_hist_avg_demand"] = (
        shifted.groupby(panel["restaurant_id"]).expanding().mean().reset_index(level=0, drop=True)
    )
    panel["restaurant_hist_volatility"] = (
        shifted.groupby(panel["restaurant_id"]).expanding().std().reset_index(level=0, drop=True)
    )

    # Zone-level demand at the same bucket, using only the PREVIOUS bucket's zone total
    # (shifted before merge) to avoid same-timestamp leakage.
    zone_ts = (
        panel.groupby(["zone_id", "bucket_ts"])["order_count"].sum().reset_index()
        .sort_values(["zone_id", "bucket_ts"])
    )
    zone_ts["zone_demand_prev_bucket"] = zone_ts.groupby("zone_id")["order_count"].shift(1)
    panel = panel.merge(
        zone_ts[["zone_id", "bucket_ts", "zone_demand_prev_bucket"]],
        on=["zone_id", "bucket_ts"], how="left",
    )
    return panel


def _add_targets(panel: pd.DataFrame, bucket_minutes: int, horizons_minutes: list[int]) -> pd.DataFrame:
    panel = panel.sort_values(["restaurant_id", "bucket_ts"]).reset_index(drop=True)
    g = panel.groupby("restaurant_id")["order_count"]
    for h in horizons_minutes:
        steps = h // bucket_minutes
        label = _minutes_to_label(h)
        panel[f"target_{label}"] = g.shift(-steps)
    return panel


def build_demand_features(raw_dir: pathlib.Path, config: dict) -> pd.DataFrame:
    orders = pd.read_parquet(raw_dir / "orders.parquet")
    weather_traffic = pd.read_parquet(raw_dir / "weather_traffic.parquet")

    bucket_minutes = config["simulation"]["bucket_minutes"]
    df_cfg = config["demand_forecasting"]

    panel = _build_demand_panel(orders, bucket_minutes)
    panel = _add_temporal_features(panel)
    panel = _add_lag_rolling_features(
        panel, bucket_minutes, df_cfg["lag_buckets"], df_cfg["rolling_windows_buckets"]
    )
    panel = _add_restaurant_zone_history(panel)

    weather_traffic = weather_traffic.rename(columns={"timestamp_hour": "hour_ts"})
    panel["hour_ts"] = panel["bucket_ts"].dt.floor("h")
    panel = panel.merge(
        weather_traffic[["zone_id", "hour_ts", "rain_mm", "weather_condition", "traffic_multiplier"]],
        on=["zone_id", "hour_ts"], how="left",
    )
    panel = panel.drop(columns=["hour_ts"])

    panel = _add_targets(panel, bucket_minutes, df_cfg["horizons_minutes"])
    return panel


def build_eta_features(raw_dir: pathlib.Path, config: dict) -> pd.DataFrame:
    orders = pd.read_parquet(raw_dir / "orders.parquet")
    restaurants = pd.read_parquet(raw_dir / "restaurants.parquet")
    weather_traffic = pd.read_parquet(raw_dir / "weather_traffic.parquet")

    orders = orders[~orders["is_cancelled"]].copy()
    orders = orders.sort_values("order_timestamp").reset_index(drop=True)

    df = orders.merge(
        restaurants[["restaurant_id", "cuisine", "popularity_score", "base_prep_time_min", "rating"]],
        on="restaurant_id", how="left",
    )

    ts = df["order_timestamp"]
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin([12, 13, 14, 19, 20, 21]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Restaurant's historical (PAST orders only) average prep time / ETA, computed via
    # expanding mean on the time-sorted series per restaurant, shifted by 1 to exclude
    # the current order itself.
    df = df.sort_values(["restaurant_id", "order_timestamp"]).reset_index(drop=True)
    grp = df.groupby("restaurant_id")
    df["restaurant_hist_avg_prep_time"] = grp["prep_time_min"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["restaurant_hist_avg_eta"] = grp["eta_minutes"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["restaurant_order_sequence_num"] = grp.cumcount()

    df = df.sort_values("order_timestamp").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default="data/raw")
    parser.add_argument("--out", type=str, default="data/processed")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = pathlib.Path(args.raw)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    demand_panel = build_demand_features(raw_dir, config)
    demand_panel.to_parquet(out_dir / "demand_timeseries.parquet", index=False)
    print(f"Demand panel: {demand_panel.shape}")

    eta_features = build_eta_features(raw_dir, config)
    eta_features.to_parquet(out_dir / "eta_features.parquet", index=False)
    print(f"ETA features: {eta_features.shape}")


if __name__ == "__main__":
    main()
