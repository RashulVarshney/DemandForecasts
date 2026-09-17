"""
SHAP-based interpretability for the final demand-forecasting and ETA models.

Produces:
  - Global feature importance bar chart (mean |SHAP value|)
  - SHAP summary (beeswarm) plot
  - A single-prediction waterfall explanation

These are called from notebooks 03/04, saving figures into reports/figures/.
Kept as a module (not notebook-only code) so the same explanation logic is
reusable and testable.
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import shap


def compute_shap_values(model, X_sample):
    explainer = shap.TreeExplainer(model)
    return explainer(X_sample)


def save_global_importance_plot(shap_values, out_path: pathlib.Path, title: str) -> None:
    plt.figure(figsize=(8, 6))
    shap.plots.bar(shap_values, show=False, max_display=15)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_summary_plot(shap_values, out_path: pathlib.Path, title: str) -> None:
    plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(shap_values, show=False, max_display=15)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_single_prediction_waterfall(shap_values, index: int, out_path: pathlib.Path, title: str) -> None:
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_values[index], show=False, max_display=12)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
