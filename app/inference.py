"""
Inference CLI for demand forecasting and ETA prediction.

Trains lightweight on-the-fly models from the processed feature tables (this
project does not persist heavyweight model binaries to keep the repo small;
see reports/model_report.md "Production considerations" for how a real
deployment would instead load a versioned model artifact from a model
registry rather than retraining per request).

Usage:
    python -m app.inference demand --restaurant_id 5 --timestamp "2026-07-01 19:30"
    python -m app.inference eta --restaurant_id 5 --distance_km 3.2 --n_items 3 \
        --basket_value_inr 450 --timestamp "2026-07-01 19:30"
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation.metrics import time_based_split
from src.forecasting.train_demand_models import _prepare_xy, TARGET_COLS
from src.eta.train_eta_models import FEATURE_COLS as ETA_FEATURE_COLS, _prepare as eta_prepare
from src.utils.config import load_config


def _fit_demand_model(panel: pd.DataFrame, target_col: str, cfg: dict):
    df, feature_cols = _prepare_xy(panel, target_col)
    train, val, _ = time_based_split(df, "bucket_ts", cfg["demand_forecasting"]["train_frac"], cfg["demand_forecasting"]["val_frac"])
    trainval = pd.concat([train, val])
    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=7, num_leaves=31,
        random_state=cfg["random_seed"], verbosity=-1,
    )
    model.fit(trainval[feature_cols], trainval[target_col])
    return model, feature_cols


def predict_demand(restaurant_id: int, timestamp: str, cfg: dict) -> dict:
    processed_dir = pathlib.Path(cfg["paths"]["processed_dir"])
    panel = pd.read_parquet(processed_dir / "demand_timeseries.parquet")

    ts = pd.Timestamp(timestamp)
    bucket_ts = ts.floor("30min")
    row = panel[(panel["restaurant_id"] == restaurant_id) & (panel["bucket_ts"] == bucket_ts)]
    if row.empty:
        raise ValueError(
            f"No feature row found for restaurant_id={restaurant_id} at bucket {bucket_ts}. "
            "The requested timestamp must fall within the generated data's date range "
            "(see data/README.md) since features rely on historical lags."
        )

    row = row.copy()
    weather_categories = sorted(panel["weather_condition"].dropna().unique())
    row["weather_code"] = pd.Categorical(row["weather_condition"], categories=weather_categories).codes

    result = {"restaurant_id": restaurant_id, "timestamp": str(ts), "bucket_ts": str(bucket_ts)}
    peak_hours = [12, 13, 14, 19, 20, 21]
    for target_col in ["target_30m", "target_1h", "target_2h"]:
        model, feature_cols = _fit_demand_model(panel, target_col, cfg)
        pred = float(model.predict(row[feature_cols])[0])
        horizon_label = target_col.replace("target_", "forecast_")
        result[horizon_label] = round(max(pred, 0), 2)

    result["expected_peak"] = bool(bucket_ts.hour in peak_hours)
    return result


def predict_eta(
    restaurant_id: int, distance_km: float, n_items: int, basket_value_inr: float,
    timestamp: str, cfg: dict,
) -> dict:
    processed_dir = pathlib.Path(cfg["paths"]["processed_dir"])
    eta_df = pd.read_parquet(processed_dir / "eta_features.parquet")
    eta_df = eta_prepare(eta_df)

    rest_hist = eta_df[eta_df["restaurant_id"] == restaurant_id].sort_values("order_timestamp")
    if rest_hist.empty:
        raise ValueError(f"No history found for restaurant_id={restaurant_id}")
    latest = rest_hist.iloc[-1]

    ts = pd.Timestamp(timestamp)
    query_row = pd.DataFrame([{
        "distance_km": distance_km,
        "n_items": n_items,
        "basket_value_inr": basket_value_inr,
        "hour": ts.hour,
        "dow": ts.dayofweek,
        "is_weekend": int(ts.dayofweek >= 5),
        "is_peak_hour": int(ts.hour in [12, 13, 14, 19, 20, 21]),
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
        "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "popularity_score": latest["popularity_score"],
        "base_prep_time_min": latest["base_prep_time_min"],
        "rating": latest["rating"],
        "traffic_multiplier": latest["traffic_multiplier"],
        "restaurant_hist_avg_prep_time": latest["restaurant_hist_avg_prep_time"],
        "restaurant_hist_avg_eta": latest["restaurant_hist_avg_eta"],
        "restaurant_order_sequence_num": latest["restaurant_order_sequence_num"] + 1,
        "zone_id": latest["zone_id"],
        "cuisine_code": latest["cuisine_code"],
        "weather_code": latest["weather_code"],
    }])[ETA_FEATURE_COLS]

    train, val, _ = time_based_split(eta_df, "order_timestamp", cfg["eta_prediction"]["train_frac"], cfg["eta_prediction"]["val_frac"])
    trainval = pd.concat([train, val])

    models = {}
    for q in cfg["eta_prediction"]["quantiles"]:
        m = lgb.LGBMRegressor(
            objective="quantile" if q != 0.5 else "regression", alpha=q,
            n_estimators=250, learning_rate=0.05, max_depth=8, num_leaves=63,
            random_state=cfg["random_seed"], verbosity=-1,
        )
        m.fit(trainval[ETA_FEATURE_COLS], trainval["eta_minutes"])
        models[q] = m

    lo_q, mid_q, hi_q = cfg["eta_prediction"]["quantiles"]
    return {
        "restaurant_id": restaurant_id,
        "timestamp": str(ts),
        "eta_minutes": round(float(models[mid_q].predict(query_row)[0]), 1),
        "lower_bound": round(float(models[lo_q].predict(query_row)[0]), 1),
        "upper_bound": round(float(models[hi_q].predict(query_row)[0]), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Food delivery demand/ETA inference")
    sub = parser.add_subparsers(dest="command", required=True)

    demand_p = sub.add_parser("demand")
    demand_p.add_argument("--restaurant_id", type=int, required=True)
    demand_p.add_argument("--timestamp", type=str, required=True)

    eta_p = sub.add_parser("eta")
    eta_p.add_argument("--restaurant_id", type=int, required=True)
    eta_p.add_argument("--distance_km", type=float, required=True)
    eta_p.add_argument("--n_items", type=int, required=True)
    eta_p.add_argument("--basket_value_inr", type=float, required=True)
    eta_p.add_argument("--timestamp", type=str, required=True)

    args = parser.parse_args()
    cfg = load_config()

    if args.command == "demand":
        result = predict_demand(args.restaurant_id, args.timestamp, cfg)
    else:
        result = predict_eta(
            args.restaurant_id, args.distance_km, args.n_items,
            args.basket_value_inr, args.timestamp, cfg,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.exit(main())
