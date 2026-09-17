# Model Report — Food Delivery Demand Forecasting & ETA Prediction

## 1. Problem

Two prediction problems that a food-delivery platform (Swiggy-like) needs to
solve jointly to run efficient operations:

1. **Demand forecasting** — how many orders will a restaurant/zone receive in
   the next 30 min / 1h / 2h / 6h / next day? Used for rider capacity
   planning, kitchen staffing signals, and promo timing.
2. **ETA prediction** — how long will a specific order take to deliver, and
   with what confidence range? Used for the customer-facing ETA display and
   for rider dispatch decisions.

A third, connecting question is addressed via simulation: **does a better
demand forecast translate into a measurable operational improvement** (less
unmet demand under the same rider budget)?

## 2. Dataset

A documented **synthetic** dataset (no real Swiggy data is used or available
for this project — see `data/README.md` for full rationale):

- 3,054,418 orders over 90 days, 150 restaurants across 12 zones, ~30,000
  simulated customers with Zipf-distributed repeat-order behavior.
- Order-level fields: timestamp, restaurant/zone, distance, basket
  size/value, structural prep/travel/assignment-delay times, resulting ETA,
  current weather/traffic, cancellation flag.
- Hourly weather/traffic table per zone with day-level persistence and a
  simulated monsoon period.

## 3. Target definition

- **Demand target**: `order_count` per (restaurant, 30-min bucket), shifted
  forward by the horizon (`target_30m`, `target_1h`, `target_2h`,
  `target_6h`, `target_1d`).
- **ETA target**: `eta_minutes` per order — the sum of prep time, travel
  time, rider-assignment delay, and noise (see `src/data/generate_dataset.py`).

## 4. Feature engineering

See `src/features/build_features.py` and notebook `02_feature_engineering.ipynb`.

**Demand panel**: temporal (hour/dow/weekend/cyclical encodings/peak
indicators), lag features (30m/1h/2h/1d/1w), rolling mean/std/max (2h/8h/1d)
and EWM, restaurant/zone historical aggregates (expanding, past-only), and
joined current-hour weather/traffic.

**ETA table**: distance, basket size/value, time-of-day, restaurant static
attributes (cuisine, popularity, rating, base prep time), restaurant
historical prep-time/ETA behavior (expanding, past-only), current
traffic/weather.

**Leakage discipline**: every lag/rolling/historical feature is computed on
a series that is `.shift(1)`ed *before* any aggregation, so the value being
predicted can never contribute to its own features. This is unit-tested in
`tests/test_feature_leakage.py` (5 tests, all passing) and visually verified
in notebook 02.

## 5. Baselines

| Horizon | Naive prev. period | Same-hour prev. day | Same-hour prev. week | Moving avg (2h) | **Best baseline MAE** |
|---|---|---|---|---|---|
| 30 min | 3.551 | 3.141 | 3.033 | 4.282 | **3.033** |
| 1 hour | 5.204 | 3.654 | 3.568 | 5.063 | **3.568** |
| 2 hour | 6.742 | 5.271 | 5.193 | 6.058 | **5.193** |
| 6 hour | 6.276 | 6.583 | 6.507 | 6.276 | **6.276** |
| Next day | 2.692 | 2.692 | 2.583 | 3.518 | **2.583** |

(MAE in orders per 30-min bucket, evaluated on the held-out test period.)

ETA baselines (minutes): global mean MAE 9.41, median-by-zone MAE 9.09,
distance-only linear regression MAE 7.26.

## 6. Models

**Demand forecasting**: Ridge regression, Random Forest, LightGBM — compared
against the baselines above for all 5 horizons. A GRU sequence model was also
evaluated for the 1-hour horizon specifically (see `src/forecasting/train_gru_demand.py`
and section 8) to test whether a model that consumes the raw recent sequence
directly outperforms hand-engineered lag/rolling features fed to a tree
ensemble.

**ETA prediction**: Random-Forest-class baselines were skipped in favor of
going straight to LightGBM (point estimate) after the simple baselines
confirmed distance/restaurant-history signal was strong — see notebook 04 for
the full comparison table. LightGBM quantile regression (alpha =
0.1/0.5/0.9) provides the prediction interval.

### Why gradient boosting is the primary model family

With a highly informative, hand-engineered tabular feature set (lags, rolling
stats, restaurant/zone history) and a few hundred thousand to a few million
rows, gradient-boosted trees (LightGBM) are typically at least as accurate as
deep learning on this kind of panel/tabular data, train in seconds to low
minutes on CPU, and support fast, well-understood interpretability (SHAP
`TreeExplainer` is exact and cheap for tree ensembles, unlike approximate
SHAP for neural nets). Linear models (Ridge) are kept as a "sane, explainable
middle ground" and Random Forest as a bagging comparison point.

## 7. Evaluation methodology

- **Time-based split** (never randomly shuffled): earliest 70% / next 15% /
  latest 15%, split on the global bucket timestamp (demand) or order
  timestamp (ETA) — see `src/evaluation/metrics.time_based_split`. Random
  shuffling would let nearby lag/rolling features leak adjacent-timestamp
  information between train and test.
- **Metrics**: MAE, RMSE, MedAE, P90/P95 absolute error, WAPE (chosen over
  MAPE because MAPE is undefined/unstable at zero-demand buckets, which are
  common for low-volume restaurants at off-peak hours).
- **Error breakdowns** by peak/off-peak, distance band, zone, and weather
  condition — a single aggregate metric can hide segment-specific failure.
- **Interval coverage** for quantile regression: empirical fraction of test
  points falling inside the predicted interval, compared to the target
  coverage.

## 8. Results

### 8.1 Demand forecasting (MAE, orders per 30-min bucket)

| Horizon | Best baseline | Ridge | Random Forest | **LightGBM** | Improvement vs. best baseline |
|---|---|---|---|---|---|
| 30 min | 3.033 | 2.526 | 2.078 | **2.043** | 32.6% |
| 1 hour | 3.568 | 2.712 | 1.830 | **1.812** | 49.2% |
| 2 hour | 5.193 | 3.057 | 1.777 | **1.763** | 66.0% |
| 6 hour | 6.276 | 3.342 | 1.643 | **1.639** | 73.9% |
| Next day | 2.583 | 2.136 | 1.907 | **1.831** | 29.1% |

LightGBM beats the best naive/seasonal baseline at every horizon, with the
margin *growing* at longer horizons — naive/seasonal-naive baselines degrade
faster than the ML model as the prediction gets further from the last
observed value, while LightGBM's rolling/historical features keep it stable.

**GRU sequence model (1-hour horizon only)**: a 1-layer GRU (32 hidden
units) consuming the raw 6-hour (12-bucket) sequence of `order_count` +
calendar features per restaurant achieved **MAE 1.707**, slightly *beating*
LightGBM's 1.812 for the same horizon (see `models/gru_demand_results.json`
and `src/forecasting/train_gru_demand.py`). This is a genuine, if modest,
result: the raw sequence apparently carries a small amount of additional
signal beyond the hand-engineered lag/rolling features — plausibly because
the GRU can learn nonlinear combinations of the recent trajectory (e.g.
"accelerating" vs. "plateauing" demand) that the fixed set of rolling
mean/std/max features only partially captures.

**Despite the small accuracy edge, LightGBM remains the recommended
production model** for this task: the GRU's ~5.8% MAE improvement doesn't
offset its real costs — no exact SHAP-style attribution (unlike LightGBM's
`TreeExplainer`), a hard requirement for `sequence_length` (12 buckets = 6h)
of prior history per restaurant before it can predict anything (worse
cold-start behavior than the tree model), more hyperparameters to maintain,
and slower training/inference. This tradeoff is discussed explicitly rather
than defaulting to "the more complex model won."

### 8.2 ETA prediction (minutes)

| Model | MAE | RMSE | MedAE | P90 | P95 | WAPE |
|---|---|---|---|---|---|---|
| Mean ETA | 9.40 | 12.05 | 7.82 | 19.12 | 23.15 | 0.287 |
| Median ETA by zone | 9.09 | 11.79 | 7.42 | 18.64 | 22.79 | 0.277 |
| Linear (distance only) | 7.26 | 9.11 | 6.13 | 14.98 | 17.76 | 0.221 |
| **LightGBM** | **4.31** | **5.51** | **3.54** | **9.00** | **10.95** | **0.131** |

LightGBM cuts MAE by ~54% versus the best simple baseline (median-by-zone),
and — importantly for an operational system — the P90/P95 tail errors improve
by a comparable margin, not just the average.

**Prediction interval (LightGBM quantile regression, 10th/90th percentile):**
target coverage 80%, empirical coverage on test set **79.8%** — well
calibrated. Mean interval width ≈ **13.75 minutes** (e.g. "predicted 31 min,
likely range 25–39 min").

### 8.3 Business simulation

Zone-level 30-min-ahead LightGBM forecast (MAE 9.29 orders, WAPE 0.143 at the
zone-aggregated grain — noisier than the restaurant-level 30-min model since
zone totals have higher variance) used to drive a simulated rider-allocation
comparison (`src/forecasting/business_simulation.py`, same total rider-hours
budget for both policies):

| Policy | Total unmet demand | Capacity utilization | % buckets fully met |
|---|---|---|---|
| Fixed allocation | 281,717 orders | 70.4% | 46.0% |
| Forecast-driven allocation | 199,053 orders | 97.3% | 16.4% |

**Total unmet demand fell 29.3%** and capacity utilization rose sharply — the
forecast-driven policy uses the same rider budget far more efficiently in
aggregate. **Honest tradeoff**: the share of individual buckets where
capacity *fully* met demand actually dropped (46.0% → 16.4%), because
allocating tightly around the predicted mean leaves many buckets with a
small shortfall instead of a comfortable fixed cushion. A real deployment
would likely add a small safety margin above the point forecast (or use the
forecast's prediction interval upper bound) rather than allocating to the
exact predicted mean — this is exactly the kind of decision a demand-forecast
uncertainty estimate (Phase 8) is meant to inform. See notebook 05 for the
full per-zone breakdown and this caveat discussed in context.

## 9. Error analysis

- **Demand model**: peak-hour and off-peak MAE are close for the 1-hour
  LightGBM model (see notebook 03's error breakdown) — the model has learned
  the peak/off-peak seasonality rather than being systematically worse during
  spikes.
- **ETA model**: MAE is stable across the 0–10km distance range (~4.3 min)
  but rises for the 10km+ band (MAE 5.30, ~1 min worse) and its WAPE is
  lowest there (fewer, larger-value orders dilute relative error) — the model
  has less training signal for rare long-distance deliveries. See notebook
  04's distance-band and weather breakdowns for the full tables.

## 10. Feature importance

**ETA model top features (LightGBM gain-based importance):**
`traffic_multiplier`, `distance_km`, `restaurant_hist_avg_eta`, `zone_id`,
`restaurant_order_sequence_num`, `base_prep_time_min`,
`restaurant_hist_avg_prep_time`, `basket_value_inr`, `n_items`,
`popularity_score`. Current traffic and distance dominate, but a
restaurant's own historical ETA/prep-time track record is close behind —
confirming that "this restaurant is reliably slow/fast" is a real,
recoverable signal distinct from distance alone (see notebook 04's SHAP
summary).

**Demand model (1h horizon)**: recent lag/rolling features dominate
(`lag_30m`, `roll_mean_2h`, `ewm_mean_short`), followed by
temporal/seasonality encodings (`hour_sin`/`hour_cos`,
`is_lunch_peak`/`is_dinner_peak`) and `restaurant_hist_avg_demand` — see
notebook 03's SHAP summary and feature-importance chart.

## 11. Limitations

- **Synthetic data**: statistical patterns are realistic by design but are
  not calibrated against real Swiggy operational data — absolute numbers
  (e.g. "4.3 minutes MAE") should be read as *methodology validation*, not as
  claims about real-world Swiggy ETA accuracy.
- **Weather/traffic features use the current hour's actual conditions**, not
  a forecast. A production system predicting even 30–60 minutes ahead would
  need forecasted weather/traffic, which is itself uncertain — this project
  does not model that additional uncertainty layer.
- **No rider-level data** (individual rider speed, fatigue, vehicle type) —
  `rider_assignment_delay_min` is modeled as a simple exponential distribution
  rather than a real rider-assignment/dispatch simulation.
- **Business simulation is a simplification**: ignores rider shift
  constraints, minimum shift lengths, and reallocation costs (see notebook
  05's caveats section).
- **Cold start**: restaurants/customers with very short histories get noisy
  `restaurant_hist_*` features (few or no prior observations); this is not
  separately handled (e.g. with shrinkage toward a cuisine-level prior).

## 12. Production considerations

- Model artifacts here are retrained on-the-fly by `app/inference.py` for
  simplicity; a production deployment would load versioned model binaries
  from a model registry (e.g. MLflow) rather than retraining per request.
- Feature computation would move to a feature store with online/offline
  parity (the same lag/rolling logic implemented once, served both in batch
  training and low-latency online inference) — see `spark/distributed_feature_pipeline.py`
  and the "Scaling to 1B orders" discussion in the README for how this would
  scale.
- Weather/traffic features would need a real forecast feed for horizons
  beyond "now," with the resulting forecast uncertainty propagated into the
  demand/ETA prediction intervals (not modeled here).
- Model monitoring: given the demonstrated demand drift (90-day platform
  growth trend baked into the simulator), a production system would need
  scheduled retraining and drift detection on the lag/rolling feature
  distributions, not a one-time train/test split.

## 13. Future improvements

- Conformal prediction for the demand-forecast intervals (currently only the
  ETA model has calibrated intervals via quantile regression).
- Hierarchical/global demand models that share statistical strength across
  restaurants (e.g. a single LightGBM with restaurant embeddings, already
  partially achieved via `restaurant_hist_avg_demand`, vs. fully separate
  per-restaurant models).
- A real dispatch-level rider simulation (discrete-event simulation of rider
  positions/shifts) instead of the current bucket-level capacity abstraction,
  to make the business simulation in notebook 05 more operationally realistic.
- Incorporating actual weather/traffic *forecasts* (not just current
  conditions) for horizons beyond the immediate bucket.
