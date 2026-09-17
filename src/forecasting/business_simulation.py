"""
Business simulation: does using demand forecasts to allocate rider capacity
actually reduce unmet demand / delay, versus a fixed static allocation?

This is a SIMULATION, not a claim about real Swiggy operations. It uses the
zone-level demand panel and the LightGBM 30-min-ahead forecast already
trained in src/forecasting/train_demand_models.py.

Mechanics
---------
For each 30-min bucket and zone:
  - `demand` = actual total completed orders in that zone/bucket (from data)
  - `capacity` = riders_allocated * rider_capacity_per_unit (orders one rider
    can serve per bucket, from config)

Two allocation policies are compared, both using the SAME total rider-hours
budget per zone across the whole test period (so this is a fair "smarter
scheduling of the same capacity" comparison, not "we hired more riders"):

  1. FIXED allocation: every bucket gets the zone's fixed_capacity_riders
     (config), i.e. today's naive "peak planning" approach.
  2. FORECAST-DRIVEN allocation: riders allocated proportionally to the
     PREDICTED demand for that bucket (using the already-trained 30-min
     LightGBM forecast), subject to the same total rider-hours budget as the
     fixed policy, so it is reallocating capacity rather than adding it.

Metrics reported per policy: total unmet demand (demand - capacity, floored
at 0, summed), capacity utilization, and service level (% of buckets where
capacity >= demand).

Run:
    python -m src.forecasting.business_simulation
"""
from __future__ import annotations

import json
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation.metrics import time_based_split
from src.utils.config import load_config


def _train_zone_forecast(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Aggregate the restaurant-level panel to zone-level demand and fit a
    LightGBM 30-min-ahead forecaster at the ZONE grain (rider allocation
    happens per zone, not per restaurant)."""
    df_cfg = cfg["demand_forecasting"]
    target_col = "target_30m"

    zone_panel = (
        panel.groupby(["zone_id", "bucket_ts"])
        .agg(order_count=("order_count", "sum"))
        .reset_index()
        .sort_values(["zone_id", "bucket_ts"])
    )
    g = zone_panel.groupby("zone_id")["order_count"]
    zone_panel["lag_30m"] = g.shift(1)
    zone_panel["lag_1h"] = g.shift(2)
    zone_panel["lag_1d"] = g.shift(48)
    zone_panel["roll_mean_2h"] = g.shift(1).groupby(zone_panel["zone_id"]).rolling(4, min_periods=1).mean().reset_index(level=0, drop=True)
    zone_panel["hour"] = zone_panel["bucket_ts"].dt.hour
    zone_panel["dow"] = zone_panel["bucket_ts"].dt.dayofweek
    zone_panel["is_weekend"] = (zone_panel["dow"] >= 5).astype(int)
    zone_panel["target_30m"] = g.shift(-1)

    feature_cols = ["lag_30m", "lag_1h", "lag_1d", "roll_mean_2h", "hour", "dow", "is_weekend"]
    df = zone_panel.dropna(subset=feature_cols + [target_col])
    train, val, test = time_based_split(df, "bucket_ts", df_cfg["train_frac"], df_cfg["val_frac"])
    trainval = pd.concat([train, val])

    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
        random_state=cfg["random_seed"], verbosity=-1,
    )
    model.fit(trainval[feature_cols], trainval[target_col])

    test = test.copy()
    test["predicted_demand"] = np.clip(model.predict(test[feature_cols]), 0, None)
    test["actual_demand"] = test[target_col]
    return test[["zone_id", "bucket_ts", "actual_demand", "predicted_demand"]]


def simulate(zone_test: pd.DataFrame, sim_cfg: dict) -> dict:
    riders_per_unit = sim_cfg["rider_capacity_per_unit"]
    fixed_riders = sim_cfg["fixed_capacity_riders"]

    summary_rows = []
    for zone_id, g in zone_test.groupby("zone_id"):
        g = g.sort_values("bucket_ts")
        n_buckets = len(g)

        # Fixed policy: same rider count every bucket.
        fixed_capacity = np.full(n_buckets, fixed_riders * riders_per_unit)
        total_rider_hours_budget = fixed_riders * n_buckets  # riders * buckets, held constant

        # Forecast-driven policy: allocate riders proportional to predicted demand,
        # normalized so total rider-bucket budget matches the fixed policy exactly.
        pred = g["predicted_demand"].values
        pred_share = pred / pred.sum() if pred.sum() > 0 else np.full(n_buckets, 1 / n_buckets)
        forecast_riders = pred_share * total_rider_hours_budget
        forecast_capacity = forecast_riders * riders_per_unit

        actual = g["actual_demand"].values

        for policy, capacity in [("fixed", fixed_capacity), ("forecast_driven", forecast_capacity)]:
            unmet = np.clip(actual - capacity, 0, None)
            served = np.minimum(actual, capacity)
            utilization = served.sum() / capacity.sum() if capacity.sum() > 0 else np.nan
            service_level = float(np.mean(capacity >= actual))
            summary_rows.append(
                {
                    "zone_id": zone_id,
                    "policy": policy,
                    "total_demand": float(actual.sum()),
                    "total_capacity": float(capacity.sum()),
                    "total_unmet_demand": float(unmet.sum()),
                    "capacity_utilization": float(utilization),
                    "service_level_pct_buckets_met": service_level,
                }
            )
    return pd.DataFrame(summary_rows)


def main():
    cfg = load_config()
    panel = pd.read_parquet(pathlib.Path(cfg["paths"]["processed_dir"]) / "demand_timeseries.parquet")

    zone_test = _train_zone_forecast(panel, cfg)
    sim_results = simulate(zone_test, cfg["business_simulation"])

    overall = sim_results.groupby("policy")[
        ["total_demand", "total_capacity", "total_unmet_demand", "capacity_utilization", "service_level_pct_buckets_met"]
    ].sum(numeric_only=False)
    # utilization/service_level should be averaged, not summed, across zones
    overall["capacity_utilization"] = sim_results.groupby("policy")["capacity_utilization"].mean()
    overall["service_level_pct_buckets_met"] = sim_results.groupby("policy")["service_level_pct_buckets_met"].mean()

    print("Per-zone simulation results (first 10 rows):")
    print(sim_results.head(10))
    print("\nOverall comparison (fixed vs forecast-driven allocation):")
    print(overall)

    fixed_unmet = overall.loc["fixed", "total_unmet_demand"]
    forecast_unmet = overall.loc["forecast_driven", "total_unmet_demand"]
    reduction_pct = 100.0 * (fixed_unmet - forecast_unmet) / fixed_unmet if fixed_unmet > 0 else float("nan")
    print(f"\nUnmet demand reduction from forecast-driven allocation: {reduction_pct:.1f}%")

    models_dir = pathlib.Path("models")
    models_dir.mkdir(exist_ok=True)
    sim_results.to_csv(models_dir / "business_simulation_per_zone.csv", index=False)
    with open(models_dir / "business_simulation_summary.json", "w") as f:
        json.dump(
            {
                "overall": overall.to_dict(orient="index"),
                "unmet_demand_reduction_pct": reduction_pct,
            },
            f, indent=2, default=str,
        )
    print(f"\nSaved to {models_dir / 'business_simulation_summary.json'}")


if __name__ == "__main__":
    main()
