"""
Train and compare ETA (delivery-time) prediction models, plus quantile
regression for prediction intervals.

Models compared:
  - Baselines: global mean, median-by-zone, simple linear regression
  - Random Forest
  - LightGBM (point estimate, objective=regression)
  - LightGBM quantile regression at 0.1 / 0.5 / 0.9 (uncertainty intervals)

Time-based split (70/15/15) on order_timestamp, same rationale as demand
forecasting: ETA behavior drifts over the 90 days (platform growth, capacity
changes), so a random split would let the model implicitly learn from
"future" traffic/restaurant conditions.

Run:
    python -m src.eta.train_eta_models
"""
from __future__ import annotations

import json
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.evaluation.metrics import (
    regression_report, time_based_split, error_breakdown,
    interval_coverage, mean_interval_width,
)
from src.utils.config import load_config

TARGET_COL = "eta_minutes"
FEATURE_COLS = [
    "distance_km", "n_items", "basket_value_inr", "hour", "dow", "is_weekend",
    "is_peak_hour", "hour_sin", "hour_cos", "popularity_score", "base_prep_time_min",
    "rating", "traffic_multiplier", "restaurant_hist_avg_prep_time",
    "restaurant_hist_avg_eta", "restaurant_order_sequence_num", "zone_id", "cuisine_code",
    "weather_code",
]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cuisine_code"] = df["cuisine"].astype("category").cat.codes
    df["weather_code"] = df["weather_condition"].astype("category").cat.codes
    # Drop rows with no restaurant history yet (first order of each restaurant)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    return df


def _baselines(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    global_mean = train[TARGET_COL].mean()
    pred_mean = np.full(len(test), global_mean)

    zone_median = train.groupby("zone_id")[TARGET_COL].median()
    pred_zone_median = test["zone_id"].map(zone_median).fillna(global_mean).values

    lin = LinearRegression()
    lin.fit(train[["distance_km"]], train[TARGET_COL])
    pred_linear = lin.predict(test[["distance_km"]])

    return {
        "mean_eta": pred_mean,
        "median_eta_by_zone": pred_zone_median,
        "linear_distance_only": pred_linear,
    }


def train_and_evaluate(df: pd.DataFrame, cfg: dict) -> dict:
    df = _prepare(df)
    train, val, test = time_based_split(df, "order_timestamp", cfg["train_frac"], cfg["val_frac"])
    trainval = pd.concat([train, val])

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_val, y_val = val[FEATURE_COLS], val[TARGET_COL]
    X_trainval, y_trainval = trainval[FEATURE_COLS], trainval[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    results = {}
    for name, preds in _baselines(train, test).items():
        results[name] = regression_report(y_test, preds)

    lgb_point = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=8, num_leaves=63,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        random_state=cfg.get("random_seed", 42), verbosity=-1,
    )
    lgb_point.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
    best_iter = lgb_point.best_iteration_ or lgb_point.n_estimators
    lgb_final = lgb.LGBMRegressor(
        n_estimators=best_iter, learning_rate=0.05, max_depth=8, num_leaves=63,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        random_state=cfg.get("random_seed", 42), verbosity=-1,
    )
    lgb_final.fit(X_trainval, y_trainval)
    test_pred_point = lgb_final.predict(X_test)
    results["lightgbm_point"] = regression_report(y_test, test_pred_point)

    # --- Quantile regression for prediction intervals ---
    quantile_preds = {}
    for q in cfg["quantiles"]:
        qmodel = lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=best_iter, learning_rate=0.05, max_depth=8, num_leaves=63,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
            random_state=cfg.get("random_seed", 42), verbosity=-1,
        )
        qmodel.fit(X_trainval, y_trainval)
        quantile_preds[q] = qmodel.predict(X_test)

    lo_q, mid_q, hi_q = cfg["quantiles"]
    coverage = interval_coverage(y_test, quantile_preds[lo_q], quantile_preds[hi_q])
    interval_width = mean_interval_width(quantile_preds[lo_q], quantile_preds[hi_q])
    results["quantile_interval"] = {
        "target_coverage": hi_q - lo_q,
        "empirical_coverage": coverage,
        "mean_interval_width_minutes": interval_width,
        "median_point_MAE_vs_q50": regression_report(y_test, quantile_preds[mid_q])["MAE"],
    }

    # Error breakdown by peak/off-peak and by distance band for the point model
    test_eval = test.copy()
    test_eval["y_pred"] = test_pred_point
    test_eval["peak_period"] = np.where(test_eval["is_peak_hour"] == 1, "peak", "off_peak")
    test_eval["distance_band"] = pd.cut(
        test_eval["distance_km"], bins=[0, 2, 5, 10, np.inf],
        labels=["0-2km", "2-5km", "5-10km", "10km+"],
    )
    peak_breakdown = error_breakdown(test_eval, TARGET_COL, "y_pred", "peak_period")
    distance_breakdown = error_breakdown(test_eval, TARGET_COL, "y_pred", "distance_band")
    zone_breakdown = error_breakdown(test_eval, TARGET_COL, "y_pred", "zone_id")
    weather_breakdown = error_breakdown(test_eval, TARGET_COL, "y_pred", "weather_condition")

    return {
        "results": results,
        "feature_cols": FEATURE_COLS,
        "peak_breakdown": peak_breakdown.to_dict(orient="records"),
        "distance_breakdown": distance_breakdown.to_dict(orient="records"),
        "zone_breakdown": zone_breakdown.to_dict(orient="records"),
        "weather_breakdown": weather_breakdown.to_dict(orient="records"),
        "feature_importance": dict(
            sorted(zip(FEATURE_COLS, lgb_final.feature_importances_.tolist()), key=lambda x: -x[1])
        ),
        "n_train": len(X_trainval),
        "n_test": len(X_test),
    }


def main():
    cfg = load_config()
    eta_cfg = {**cfg["eta_prediction"], "random_seed": cfg["random_seed"]}
    df = pd.read_parquet(pathlib.Path(cfg["paths"]["processed_dir"]) / "eta_features.parquet")

    out = train_and_evaluate(df, eta_cfg)
    leaderboard = pd.DataFrame(
        {k: v for k, v in out["results"].items() if k != "quantile_interval"}
    ).T
    print(leaderboard)
    print("\nQuantile interval performance:")
    print(out["results"]["quantile_interval"])
    print("\nPeak vs off-peak error breakdown:")
    print(pd.DataFrame(out["peak_breakdown"]))
    print("\nDistance-band error breakdown:")
    print(pd.DataFrame(out["distance_breakdown"]))

    models_dir = pathlib.Path("models")
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "eta_prediction_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved results to {models_dir / 'eta_prediction_results.json'}")


if __name__ == "__main__":
    main()
