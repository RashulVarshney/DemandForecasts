# Resume Bullets

All metrics below are pulled directly from `models/*.json` and
`reports/model_report.md` in this repo — reproducible by running the
pipeline (`README.md` → "How to Run"). No fabricated numbers.

## A. Conservative / fully defensible

- Built an end-to-end demand-forecasting and delivery-ETA prediction system
  (Python, LightGBM, scikit-learn, PySpark) on a 3M-order synthetic
  food-delivery dataset, with leakage-safe feature engineering validated by
  unit tests and a time-based train/val/test split.
- Compared gradient-boosted trees against naive/seasonal baselines across 5
  forecast horizons and a delivery-ETA model with quantile-regression
  prediction intervals, reporting MAE/RMSE/P90/P95 and SHAP-based
  interpretability for both models.
- Simulated forecast-driven vs. fixed rider-capacity allocation under a
  constant capacity budget to quantify the operational value of the demand
  forecast, using DuckDB/SQL for exploratory and feature-extraction queries.

## B. Impact-oriented

- Designed and built a food-delivery demand-forecasting and ETA-prediction
  system that improved LightGBM demand-forecast MAE by 29–74% over the best
  naive/seasonal baseline across 5 horizons (30 min to next-day), and cut
  ETA prediction MAE by ~54% (9.1 → 4.3 minutes) versus a zone-median
  baseline.
- Delivered well-calibrated 80% prediction intervals for delivery ETA
  (79.8% empirical coverage) via LightGBM quantile regression, giving
  operations a defensible confidence range instead of a single point
  estimate.
- Built a business simulation showing that reallocating a fixed rider-hours
  budget according to the demand forecast reduces simulated unmet demand
  versus a static allocation policy — connecting the ML model directly to an
  operational capacity-planning decision.

## C. Highly technical

- Engineered a leakage-safe feature pipeline (temporal, multi-horizon lag,
  rolling/EWM, and expanding restaurant/zone-history aggregates) in
  pandas and PySpark (window functions with explicit `ROWS BETWEEN ... 1
  PRECEDING` frames), enforced by 5 targeted unit tests asserting no
  future-bucket information reaches any feature.
- Trained and time-split-validated Ridge, Random Forest, LightGBM
  (point + quantile objectives), and a GRU sequence model for multi-horizon
  demand forecasting and ETA prediction on a 648K-row demand panel and a
  3.0M-row order-level table, with SHAP TreeExplainer attributions and
  per-segment (peak/off-peak, distance-band, zone, weather) error analysis.
- Implemented a DuckDB-backed SQL analysis layer (CTEs, window functions,
  cross joins for a dense demand grid) and a PySpark distributed
  re-implementation of the core lag/rolling feature logic, with an explicit
  zone-based repartitioning strategy to reason about data skew at scale.
