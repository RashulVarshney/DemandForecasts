"""
Train and compare demand-forecasting models across multiple horizons.

Models compared per horizon:
  - Baselines (naive, seasonal-naive day/week, moving average)
  - Ridge regression (linear, regularized)
  - Random Forest
  - LightGBM (gradient boosting)

A deep-learning model (GRU, see src/forecasting/train_lstm_demand.py) is
evaluated separately for the primary 1-hour horizon only -- see that module's
docstring for the justification of why/where a sequence model is tried.

Time-based split: earliest 70% train / next 15% val / latest 15% test,
split independently per-restaurant-is NOT needed here since bucket_ts is a
shared global grid -- we split on the global bucket_ts to guarantee no
restaurant's test period overlaps another's training period in time.

Run:
    python -m src.forecasting.train_demand_models
"""
from __future__ import annotations

import json
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.evaluation.metrics import regression_report, time_based_split, error_breakdown
from src.forecasting.baselines import run_all_baselines
from src.utils.config import load_config

NON_FEATURE_COLS = {
    "restaurant_id", "zone_id", "bucket_ts", "order_count", "weather_condition",
}
TARGET_COLS = ["target_30m", "target_1h", "target_2h", "target_6h", "target_1d"]


def _prepare_xy(panel: pd.DataFrame, target_col: str):
    feature_cols = [
        c for c in panel.columns
        if c not in NON_FEATURE_COLS and c not in TARGET_COLS
    ]
    df = panel.dropna(subset=[target_col]).copy()
    # Encode weather condition as a category code (present in feature set via one-hot-ish signal)
    df["weather_code"] = df["weather_condition"].astype("category").cat.codes
    feature_cols = feature_cols + ["weather_code"]
    # Drop rows where early lag/rolling history is still NaN (start of each restaurant's series)
    df = df.dropna(subset=feature_cols)
    return df, feature_cols


def train_and_evaluate_horizon(panel: pd.DataFrame, target_col: str, cfg: dict) -> dict:
    df, feature_cols = _prepare_xy(panel, target_col)
    train, val, test = time_based_split(df, "bucket_ts", cfg["train_frac"], cfg["val_frac"])

    X_train, y_train = train[feature_cols], train[target_col]
    X_val, y_val = val[feature_cols], val[target_col]
    X_test, y_test = test[feature_cols], test[target_col]
    # Use train+val for fitting final models (val was only for early stopping / hp choice)
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    results = {"n_train": len(X_trainval), "n_test": len(X_test)}

    # --- Baselines (evaluated on the test set only) ---
    baseline_preds = run_all_baselines(test, target_col)
    for name, preds in baseline_preds.items():
        results[name] = regression_report(y_test, preds)

    # --- Ridge regression ---
    ridge = Ridge(alpha=1.0, random_state=cfg.get("random_seed", 42))
    ridge.fit(X_trainval, y_trainval)
    results["ridge_regression"] = regression_report(y_test, ridge.predict(X_test))

    # --- Random Forest ---
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        n_jobs=-1, random_state=cfg.get("random_seed", 42),
    )
    rf.fit(X_trainval, y_trainval)
    results["random_forest"] = regression_report(y_test, rf.predict(X_test))

    # --- LightGBM ---
    lgb_model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=7, num_leaves=31,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=cfg.get("random_seed", 42), verbosity=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    best_iter = lgb_model.best_iteration_ or lgb_model.n_estimators
    lgb_final = lgb.LGBMRegressor(
        n_estimators=best_iter, learning_rate=0.05, max_depth=7, num_leaves=31,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=cfg.get("random_seed", 42), verbosity=-1,
    )
    lgb_final.fit(X_trainval, y_trainval)
    results["lightgbm"] = regression_report(y_test, lgb_final.predict(X_test))

    # Error breakdown for the best model (LightGBM) by peak vs off-peak
    test_eval = test.copy()
    test_eval["y_pred_lgb"] = lgb_final.predict(X_test)
    test_eval["peak_period"] = np.where(
        test_eval["is_lunch_peak"] | test_eval["is_dinner_peak"], "peak", "off_peak"
    )
    breakdown = error_breakdown(test_eval, target_col, "y_pred_lgb", "peak_period")

    return {
        "target": target_col,
        "results": results,
        "feature_cols": feature_cols,
        "lgb_error_breakdown": breakdown.to_dict(orient="records"),
        "lgb_feature_importance": dict(
            sorted(zip(feature_cols, lgb_final.feature_importances_.tolist()), key=lambda x: -x[1])
        ),
        "_models": {"ridge": ridge, "random_forest": rf, "lightgbm": lgb_final},
        "_test_eval": test_eval,
    }


def main():
    cfg = load_config()
    df_cfg = cfg["demand_forecasting"]
    panel = pd.read_parquet(pathlib.Path(cfg["paths"]["processed_dir"]) / "demand_timeseries.parquet")

    all_results = {}
    for target_col in TARGET_COLS:
        print(f"\n=== Training models for {target_col} ===")
        out = train_and_evaluate_horizon(panel, target_col, {**df_cfg, "random_seed": cfg["random_seed"]})
        out.pop("_models")
        out.pop("_test_eval")
        all_results[target_col] = out
        leaderboard = pd.DataFrame(out["results"]).T[["MAE", "RMSE", "WAPE"]].sort_values("MAE")
        print(leaderboard)

    models_dir = pathlib.Path("models")
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "demand_forecasting_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to {models_dir / 'demand_forecasting_results.json'}")


if __name__ == "__main__":
    main()
