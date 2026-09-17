"""
GRU sequence model for the 1-hour-ahead demand forecast, evaluated against
the LightGBM result from `train_demand_models.py` for the SAME horizon.

Why try a sequence model at all: gradient-boosted trees see each row's
lag/rolling features independently and cannot learn cross-lag temporal
patterns (e.g. "demand accelerating over the last 3 buckets" vs. "demand
flat") except through whatever the rolling/std features happen to encode by
hand. A GRU consumes the raw recent sequence directly and can in principle
learn such patterns itself.

Why it's evaluated here rather than assumed superior: with ~650K rows and a
small number of informative hand-engineered features, tree ensembles are
usually hard to beat on tabular/panel data, and a sequence model adds real
costs -- more hyperparameters, slower training/inference, harder to explain
to stakeholders (no SHAP-style clean attribution), and a cold-start problem
for restaurants with short histories (need `sequence_length` prior buckets).
This script quantifies the actual gap so the choice of LightGBM as the
production model (see reports/model_report.md) is a measured decision, not
an assumption.

Sequence construction: for each restaurant, a sliding window of the past
`sequence_length` buckets' order_count (+ hour/dow encodings) predicts
`target_1h`. Same time-based 70/15/15 split as the tree models.

Run:
    python -m src.forecasting.train_gru_demand
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import regression_report, time_based_split
from src.utils.config import load_config

SEQUENCE_LENGTH = 12  # 12 * 30min = 6 hours of history per sample


class GRUDemandModel(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.head(h_n[-1]).squeeze(-1)


def _build_sequences(panel: pd.DataFrame, seq_len: int):
    panel = panel.sort_values(["restaurant_id", "bucket_ts"]).reset_index(drop=True)
    panel["hour_sin"] = np.sin(2 * np.pi * panel["bucket_ts"].dt.hour / 24)
    panel["hour_cos"] = np.cos(2 * np.pi * panel["bucket_ts"].dt.hour / 24)
    panel["is_weekend"] = (panel["bucket_ts"].dt.dayofweek >= 5).astype(float)

    feature_cols = ["order_count", "hour_sin", "hour_cos", "is_weekend"]
    X_list, y_list, ts_list = [], [], []
    for _, g in panel.groupby("restaurant_id"):
        values = g[feature_cols].values.astype(np.float32)
        target = g["order_count"].values.astype(np.float32)
        bucket_ts = g["bucket_ts"].values
        # target at step i+seq_len+2 corresponds to +1h ahead (2 buckets of 30min)
        for i in range(len(g) - seq_len - 2):
            X_list.append(values[i:i + seq_len])
            y_list.append(target[i + seq_len + 2])
            ts_list.append(bucket_ts[i + seq_len + 2])
    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)
    ts = pd.to_datetime(ts_list)
    return X, y, ts


def train_gru(panel: pd.DataFrame, cfg: dict, epochs: int = 8, batch_size: int = 512) -> dict:
    torch.manual_seed(cfg["random_seed"])
    X, y, ts = _build_sequences(panel, SEQUENCE_LENGTH)

    order = np.argsort(ts.values)
    X, y, ts = X[order], y[order], ts[order]
    n = len(X)
    train_end = int(n * cfg["demand_forecasting"]["train_frac"])
    val_end = int(n * (cfg["demand_forecasting"]["train_frac"] + cfg["demand_forecasting"]["val_frac"]))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # Standardize the order_count channel using TRAIN statistics only.
    mean, std = X_train[..., 0].mean(), X_train[..., 0].std() + 1e-6
    for arr in (X_train, X_val, X_test):
        arr[..., 0] = (arr[..., 0] - mean) / std
    y_train_n, y_val_n, y_test_n = (y_train - mean) / std, (y_val - mean) / std, (y_test - mean) / std

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_n)),
        batch_size=batch_size, shuffle=True,
    )
    val_X, val_y = torch.from_numpy(X_val), torch.from_numpy(y_val_n)

    model = GRUDemandModel(n_features=X.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_X)
            val_loss = loss_fn(val_pred, val_y).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch+1}/{epochs}  val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred_n = model(torch.from_numpy(X_test)).numpy()
    test_pred = test_pred_n * std + mean
    test_pred = np.clip(test_pred, 0, None)

    return {
        "gru_target_1h": regression_report(y_test, test_pred),
        "n_train": int(train_end),
        "n_test": int(n - val_end),
        "sequence_length_buckets": SEQUENCE_LENGTH,
    }


def main():
    cfg = load_config()
    panel = pd.read_parquet(pathlib.Path(cfg["paths"]["processed_dir"]) / "demand_timeseries.parquet")
    result = train_gru(panel, cfg)
    print(result)

    # Compare against the LightGBM result for the same horizon, if available.
    lgb_results_path = pathlib.Path("models/demand_forecasting_results.json")
    if lgb_results_path.exists():
        with open(lgb_results_path) as f:
            lgb_results = json.load(f)
        lgb_mae = lgb_results["target_1h"]["results"]["lightgbm"]["MAE"]
        gru_mae = result["gru_target_1h"]["MAE"]
        print(f"\nLightGBM target_1h MAE: {lgb_mae:.3f}")
        print(f"GRU target_1h MAE: {gru_mae:.3f}")
        print(f"GRU vs LightGBM: {'better' if gru_mae < lgb_mae else 'worse'} by {abs(gru_mae - lgb_mae):.3f}")

    models_dir = pathlib.Path("models")
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "gru_demand_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    main()
