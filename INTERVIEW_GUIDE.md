# Interview Guide

50+ questions an interviewer could ask about this project, organized by
category. For each: what's being tested, a strong answer, a common wrong
answer, and follow-ups. Grounded in the actual implementation in this repo —
file references are given so you can re-verify any number before an
interview.

---

## 1. Business / Problem Formulation

**Q1. Why predict demand at 30-min/1h/2h/6h/next-day horizons instead of just one horizon?**
- *Tests:* whether you understand that different horizons serve different decisions.
- *Strong answer:* short horizons (30m/1h) drive real-time rider dispatch; medium (2h/6h) drive shift planning within a day; next-day drives staffing/inventory decisions made the night before. Each has a different acceptable error tolerance and available feature set (longer horizons rely more on seasonality, less on recent momentum).
- *Common wrong answer:* "more horizons = more accurate" — horizons don't improve each other's accuracy; they serve different consumers.
- *Follow-up:* Which horizon had the best relative improvement over baseline, and why? (6h: 73.9% — naive/seasonal baselines degrade fastest at longer horizons while LightGBM's rolling stats stay informative.)

**Q2. Why build both a demand model and an ETA model instead of one combined model?**
- *Tests:* problem decomposition instinct.
- *Strong answer:* they answer different questions for different consumers (ops planning vs. customer-facing ETA) at different grains (zone/restaurant-bucket vs. individual order) with different feature availability. Combining them would force a shared grain that fits neither well.
- *Wrong answer:* "one model is simpler to maintain" — ignores that the targets have different statistical properties (count data vs. continuous duration).
- *Follow-up:* How do they connect? (The demand forecast feeds the business simulation's rider allocation, which is the ETA model's implicit capacity assumption.)

**Q3. What operational decision does this system actually improve, concretely?**
- *Strong answer:* rider capacity allocation across zones/time-buckets — see `src/forecasting/business_simulation.py` and notebook 05, which measures unmet-demand reduction under a fixed rider-hours budget.
- *Wrong answer:* vague "better customer experience" with no measurable mechanism.

**Q4. Why not just always over-provision riders to guarantee low ETAs?**
- *Strong answer:* rider-hours are a cost; over-provisioning wastes capacity in off-peak buckets while still under-provisioning at true peaks if the total budget is fixed — this is exactly the tradeoff the business simulation quantifies (fixed vs. forecast-driven allocation of the *same* budget).

**Q5. How would you define success for this system with a product/ops stakeholder?**
- *Strong answer:* service-level agreement (% of orders delivered within X minutes of predicted ETA), unmet-demand rate at peak, and rider utilization — not model MAE in isolation. Model MAE is a means to those ends.

---

## 2. Dataset

**Q6. Why is this dataset synthetic, and why should I trust results built on it?**
- *Strong answer:* no real Swiggy order-level dataset is public; using a documented, realistic simulator (`src/data/generate_dataset.py`) lets me demonstrate the full methodology — leakage prevention, time-based validation, baseline comparison, uncertainty quantification — which transfers directly to real data. The absolute metrics (e.g. 4.3 min ETA MAE) are methodology validation, not a claim about real Swiggy performance. See `data/README.md`.
- *Wrong answer:* claiming or implying it's real data, or being unable to say clearly it's synthetic.
- *Follow-up:* What would you need to validate this differently on real data? (Recalibrate every distributional assumption — peak timing, distance distribution, cancellation causes — against real order logs; the *pipeline* would not need to change.)

**Q7. What realistic structure did you build into the generator, and why?**
- *Strong answer:* lunch/dinner peaks, weekend uplift, a monsoon-season block with elevated rain affecting traffic, a platform-growth trend, restaurant-level popularity heterogeneity (log-normal), and Zipf-distributed repeat-customer behavior — each chosen to create realistic, non-trivial signal for the corresponding feature to recover (e.g. temporal features should recover the peaks; weather/traffic features should recover the monsoon effect on ETA).

**Q8. How many rows, and why does scale matter here?**
- *Strong answer:* 3.05M orders / 648K demand-panel rows. At toy scale (a few thousand rows), model comparisons are noisy and don't stress leakage bugs (e.g. small data can hide an index-alignment bug that a larger, higher-cardinality panel would expose).

**Q9. What's the customer_id Zipf distribution for, and where does it show up in analysis?**
- *Strong answer:* models realistic repeat-customer skew (a few customers order far more often) rather than every customer being unique — used in the "repeat customers" SQL query (`sql/feature_queries.sql`), which would be meaningless with uniform one-time customers.

**Q10. What fields would leak if used naively, and how did you catch that?**
- *Strong answer:* `prep_time_min`, `travel_time_min`, `rider_assignment_delay_min` are the structural components used to build `eta_minutes` — they're outcomes, not features, and are documented as such in the data dictionary (notebook 01) and excluded from `FEATURE_COLS` in `src/eta/train_eta_models.py`.

---

## 3. SQL

**Q11. Walk me through the repeat-customers query.**
- *Strong answer* (see `sql/feature_queries.sql` §1): CTE aggregates order count per customer, then classifies one-time vs. repeat and computes each group's share of total orders using a window `SUM(...) OVER ()` for the denominator — avoids a second pass/self-join for the total.

**Q12. Why use `time_bucket`/window functions instead of a GROUP BY + self-join for lag features?**
- *Strong answer:* window functions (`LAG`, `AVG(...) OVER (... ROWS BETWEEN ...)`) avoid the O(n²) cost and awkwardness of self-joining a table to itself on "previous row," and express the intent (ordered, partitioned access to neighboring rows) directly.

**Q13. In the rolling-8h query, why `ROWS BETWEEN 16 PRECEDING AND 1 PRECEDING` and not `... AND CURRENT ROW`?**
- *Strong answer:* `AND CURRENT ROW` would include the bucket being predicted in its own rolling average — leakage. `1 PRECEDING` as the frame end excludes it, mirroring the pandas `.shift(1)` before `.rolling()` pattern.

**Q14. Why WAPE over MAPE for the demand queries/metrics?**
- *Strong answer:* MAPE divides by `y_true`, which is frequently 0 for low-volume restaurant-buckets — undefined/exploding. WAPE (`sum(|error|) / sum(|actual|)`) is well-defined and just as interpretable as a normalized error. See `src/evaluation/metrics.py::wape`.

**Q15. How would you validate a SQL query is leakage-safe just by reading it?**
- *Strong answer:* check that every window frame referencing "history" ends at `1 PRECEDING` (or uses `LAG` with offset ≥ 1), never `CURRENT ROW` or `UNBOUNDED FOLLOWING`, for any column derived from the same event being predicted.

**Q16. Why `NTILE` needed a CTE instead of being used directly in `GROUP BY`?**
- *Strong answer:* `NTILE` is a window function; SQL doesn't allow window functions inside `GROUP BY` because grouping happens logically before window functions are evaluated — you must materialize the window function's output in a subquery/CTE first, then group on that result column. (This was an actual bug caught while validating `sql/exploratory_queries.sql` — see git history.)

---

## 4. EDA

**Q17. What did the hourly demand analysis reveal, and how did that shape features?**
- *Strong answer:* clear lunch (~13:00) and dinner (~20:00) peaks with a smaller breakfast bump — directly motivated the `is_lunch_peak`/`is_dinner_peak` binary features and the cyclical `hour_sin`/`hour_cos` encoding (so hour 23 and hour 0 are "close" to the model).

**Q18. Why cyclical encoding for hour/day-of-week instead of just using the raw integer?**
- *Strong answer:* a raw integer implies hour 23 and hour 0 are maximally far apart, which is false — sine/cosine encoding preserves the circular adjacency.

**Q19. What did the distance-vs-ETA relationship look like, and was it linear?**
- *Strong answer:* roughly linear in the bulk of the distribution but with rising variance at longer distances (10km+ band shows higher MAE for the trained model too — less data, more variance) — see notebook 04's distance-band breakdown.

**Q20. How did you decide which plots to include vs. skip?**
- *Strong answer:* each plot in the notebooks answers a specific modeling or business question (e.g. "does weather affect ETA enough to be a feature?") rather than being included for completeness — avoids the "dozens of meaningless plots" anti-pattern.

---

## 5. Feature Engineering

**Q21. Explain the exact mechanism that prevents lag/rolling leakage.**
- *Strong answer:* the underlying series is `.shift(1)` *before* any `.rolling()`/`.expanding()` call (`src/features/build_features.py::_add_lag_rolling_features`), so a rolling window ending at "the current row" in shifted-series terms actually only covers buckets strictly before the prediction timestamp. Verified in `tests/test_feature_leakage.py`.

**Q22. Why did you choose lag buckets of 1/2/4/48/336 (in 30-min units) specifically?**
- *Strong answer:* they correspond to human-interpretable horizons — 30min, 1h, 2h, 1 day, 1 week — matching both the target horizons and typical demand-cycle lengths (daily and weekly seasonality).

**Q23. Why use `restaurant_hist_avg_demand` as an expanding mean rather than a fixed lookback window?**
- *Strong answer:* an expanding (all-history-so-far) mean gives a stable "baseline popularity" estimate that doesn't get noisy for restaurants with sparse recent activity, complementing the fixed-window rolling features which capture recent momentum.

**Q24. Why is current-hour weather/traffic acceptable as a feature but future weather is not?**
- *Strong answer:* current conditions are observable in real time at prediction time (e.g. via a live traffic API); future weather requires its own forecast with its own error, which this project doesn't model — documented as a limitation in `reports/model_report.md`.

**Q25. How would restaurant cold-start (new restaurant, no history) be handled?**
- *Strong answer:* currently not specially handled — `restaurant_hist_*` features are NaN and those rows are dropped in `_prepare_xy`/`_prepare`. A production system would backfill with a cuisine-level or zone-level prior instead of dropping.

**Q26. Why include `restaurant_order_sequence_num` as an ETA feature?**
- *Strong answer:* it's a proxy for "how much history do we have on this restaurant" — helps the model implicitly discount the reliability of `restaurant_hist_avg_eta` for restaurants with few prior orders.

---

## 6. Statistics

**Q27. Why WAPE and P90/P95 instead of just MAE/RMSE everywhere?**
- *Strong answer:* MAE hides tail behavior that matters operationally (a rider-dispatch system cares about worst-case delay, not just average); WAPE normalizes for scale/zero-inflation better than MAPE. RMSE is kept because it penalizes large errors more, useful for catching model instability MAE would mask.

**Q28. What does "empirical coverage 79.8% vs. target 80%" tell you statistically?**
- *Strong answer:* the quantile regression intervals are well-calibrated on the test distribution — not systematically over- or under-confident. A large gap (e.g. 60% empirical vs. 80% target) would indicate the model is overconfident (intervals too narrow).

**Q29. Why is the Zipf/power-law customer distribution relevant statistically?**
- *Strong answer:* it produces a long-tailed, non-uniform distribution realistic of real customer behavior, which matters for anyone building customer-level features (e.g. customer lifetime value) even though this project doesn't build customer-level models itself.

**Q30. Why min_periods=1 in the rolling-window calls?**
- *Strong answer:* without it, the first `window-1` rows of each restaurant's series would be NaN even when *some* history exists (e.g. 1 prior bucket for a 4-bucket window) — `min_periods=1` computes a partial-window statistic instead of discarding usable partial history. Tradeoff: early estimates are noisier (based on fewer buckets).

---

## 7. Machine Learning

**Q31. Why LightGBM over XGBoost or plain scikit-learn GradientBoostingRegressor?**
- *Strong answer:* LightGBM's histogram-based split finding is substantially faster on datasets this size (450K–3M rows) with comparable accuracy to XGBoost, and its native quantile-regression objective (`objective="quantile"`) made the prediction-interval work (Phase 8) straightforward without a separate library.

**Q32. Why not tune hyperparameters with full grid/Bayesian search?**
- *Strong answer:* reasonable defaults plus early stopping on the validation set (see `lgb.early_stopping` in `train_demand_models.py`/`train_eta_models.py`) were sufficient to clearly beat baselines; exhaustive tuning would improve the numbers marginally but wasn't the point of this project — the emphasis is on methodology (leakage prevention, time-based validation, baseline comparison) over squeezing out the last percent of accuracy.

**Q33. Why does Random Forest MAE track LightGBM so closely for demand but ETA only uses LightGBM?**
- *Strong answer:* for the demand panel, the dominant signal (recent lag/rolling values) is captured well by either bagging or boosting; I chose to also run Random Forest for the demand comparison specifically to demonstrate breadth of model comparison, and reserved deeper effort (quantile regression, extensive breakdown) for LightGBM as the higher-value use of my time on the ETA side where tail behavior matters more.

**Q34. What would overfitting look like here, and how would you detect it?**
- *Strong answer:* train MAE much lower than validation/test MAE, or feature importance dominated by a near-duplicate-of-target feature (e.g. if `lag_30m` for the 30m target accidentally weren't shifted). Early stopping on a genuinely time-separated validation set is the primary guard; the leakage unit tests are a secondary guard against a *specific* class of overfitting (temporal leakage) that a train/val gap alone wouldn't necessarily catch if both were contaminated the same way.

**Q35. Why Ridge (L2) rather than Lasso (L1) as the linear baseline?**
- *Strong answer:* the feature set doesn't need aggressive sparsity/feature-selection (all features are meaningfully engineered, not a high-dimensional raw feature blow-up); Ridge's smooth shrinkage is a fairer "explainable baseline" comparison than Lasso's harsher zeroing-out.

---

## 8. Deep Learning

**Q36. Why try a GRU at all instead of relying on gradient boosting from the start?**
- *Strong answer:* to test, not assume, whether a sequence model can learn cross-lag temporal patterns (e.g. "demand accelerating over 3 buckets") that hand-engineered lag/rolling features might not fully capture — see the docstring/justification in `src/forecasting/train_gru_demand.py`.

**Q37. Why GRU over LSTM?**
- *Strong answer:* GRU has fewer parameters (no separate cell state) and trains faster with comparable performance to LSTM on short-to-medium sequences like the 12-bucket (6-hour) window used here — a reasonable default when there's no strong prior favoring LSTM's extra gating.

**Q38. What's the actual GRU vs. LightGBM result, and what does it imply?**
- *Strong answer:* see `models/gru_demand_results.json` — [fill in from your run]. If LightGBM wins (typical for this data regime), that's evidence tabular gradient boosting is the right production choice for this feature set; if GRU wins, it suggests the raw sequence carries information the hand-engineered features miss, worth investing further DL effort.

**Q39. What are the practical costs of choosing the GRU even if its accuracy were comparable?**
- *Strong answer:* slower training/inference, more hyperparameters (hidden size, sequence length, learning rate, epochs), no clean SHAP-style attribution, and a cold-start requirement (needs `sequence_length` prior buckets per restaurant, worse than a tree model's per-row independence).

**Q40. How did you prevent leakage in the GRU's train/val/test split given it uses sliding windows?**
- *Strong answer:* sequences are built per-restaurant in time order, then all sequences are globally sorted by their target timestamp and split by the same 70/15/15 time boundaries as the tree models — see `train_gru_demand.py::train_gru`. Normalization (mean/std) is fit on the train split only and applied to val/test, avoiding a second leakage channel (target-scaling leakage).

---

## 9. Time Series

**Q41. Why is a global time split (by timestamp) used instead of splitting per-restaurant?**
- *Strong answer:* a shared bucket grid means every restaurant is observed at the same timestamps; splitting per-restaurant could let restaurant A's "test" period overlap restaurant B's "train" period in wall-clock time, and since some features (zone-level demand) mix across restaurants, that would still leak indirectly.

**Q42. How do you know the 90-day period doesn't hide a distribution shift the model can't handle?**
- *Strong answer:* the simulator has a deliberate ~25% platform-growth trend and a monsoon block mid-period — the fact that LightGBM still clearly beats baselines on the final 15% (which includes the post-monsoon recovery period) is evidence the model generalizes across the regime change, not just within one seasonal window.

**Q43. Why not use classical time-series models (ARIMA/Prophet/ETS)?**
- *Strong answer:* those excel for single or few series with strong univariate seasonality; here there are 150 restaurants × 12 zones with cross-sectional features (restaurant popularity, current weather) that a panel/tabular ML approach naturally incorporates, whereas fitting 150 separate ARIMA models per restaurant would ignore that shared structure and wouldn't easily use exogenous regressors like traffic.

**Q44. What's the risk of using `roll_mean_2h` as a substitute lag for the 6h horizon baseline?**
- *Strong answer:* it's an approximation (no true "6h-ago" lag bucket was configured) — `BASELINE_LAG_FOR_HORIZON` documents this explicitly as a "nearest sensible proxy" rather than silently using a wrong or misleading baseline.

---

## 10. Model Evaluation

**Q45. Walk through exactly how you'd know if LightGBM's improvement is "real" and not noise.**
- *Strong answer:* the improvement over baseline is large and consistent across all 5 independent horizons (29–74%), not a single narrow comparison — a spurious result would be unlikely to show up so consistently across differently-constructed targets.

**Q46. Why report P90/P95 error in addition to MAE for ETA specifically?**
- *Strong answer:* ETA is customer- and rider-facing; a model that's right on average but wildly wrong 10% of the time creates a worse experience than a slightly-higher-average model with tighter tails. P90/P95 make that tradeoff visible where MAE alone would hide it.

**Q47. How do you validate the quantile regression intervals aren't just wide-and-lazy?**
- *Strong answer:* check both coverage AND width together — 79.8% empirical coverage against an 80% target with a 13.75-minute average width is meaningfully informative (not e.g. "0–120 minutes," which would trivially achieve high coverage while being useless).

---

## 11. Leakage

**Q48. Give a concrete example of a leakage bug you could imagine making here, and how the tests would catch it.**
- *Strong answer:* if `_add_lag_rolling_features` accidentally computed rolling stats on the unshifted `order_count` series (forgetting `.shift(1)` first), `test_rolling_mean_excludes_current_bucket` would fail immediately because it asserts the rolling value equals the mean of *strictly preceding* buckets, not a window including the current one.

**Q49. Beyond lag/rolling features, what's a subtler leakage risk in this project?**
- *Strong answer:* using current-hour weather/traffic as a feature is *not* leakage for "now" predictions but would become leakage if applied naively to a forecast further in the future without accounting for weather-forecast uncertainty — flagged explicitly as a documented limitation rather than silently ignored.

**Q50. Why is random train/test shuffling specifically dangerous for the ETA model, given each order is a single row (not a time series in the usual sense)?**
- *Strong answer:* the `restaurant_hist_avg_eta`/`restaurant_hist_avg_prep_time` features are expanding averages over each restaurant's order history in time order — a random shuffle would randomly interleave test-period orders throughout the "historical average" computation, meaning some test rows' features would already include information from other test-period orders. Time-based splitting prevents this.

---

## 12. Production

**Q51. How would you actually deploy this?**
- *Strong answer:* versioned model artifacts in a registry (not retrained per request, unlike the current `app/inference.py` simplification, documented as such), a feature store providing online/offline parity for the lag/rolling features, and a monitoring layer tracking both prediction drift and the input feature distributions (given the demonstrated demand trend/seasonality shifts).

**Q52. Why does `app/inference.py` retrain models on every call — isn't that a red flag?**
- *Strong answer:* it's an explicit, documented simplification for this portfolio project to avoid persisting/loading large model binaries in the repo; the report calls this out directly under "Production considerations" rather than presenting it as production-ready.

**Q53. How would you monitor this system in production?**
- *Strong answer:* track live MAE/P90 against a rolling window of ground truth (once actual ETA/demand is observed), input feature drift (e.g. distribution of `distance_km`, `traffic_multiplier` vs. training), and interval coverage drift for the quantile model — a coverage drop signals miscalibration before raw MAE necessarily moves.

---

## 13. Spark

**Q54. Why does the Spark pipeline duplicate the pandas feature logic instead of just running pandas at scale?**
- *Strong answer:* it demonstrates the same leakage-safe window-function logic expressed in a distributed engine (`Window.partitionBy(...).orderBy(...).rowsBetween(...)`), which is what a real multi-node deployment (data too large for one machine's memory) would need — pandas doesn't scale horizontally.

**Q55. Why repartition by `zone_id` before the window operations?**
- *Strong answer:* downstream consumers (zone-level aggregates, rider allocation) operate per zone, so co-locating a zone's rows on one partition avoids an extra shuffle — though this is called out as needing salting on a real cluster if a few zones are much busier than others (data skew).

**Q56. How would this scale from 1M to 1B orders?**
- *Strong answer (see README's dedicated section):* partition Parquet by date/zone for pruning, use Spark's window functions instead of pandas groupby, adopt a feature store for consistent online/offline features, add incremental/streaming computation for lag features instead of full recomputation, watch for partition skew on popular zones, and separate model training (batch, Spark/Ray) from serving (low-latency, feature-store lookups + a lightweight model runtime).

---

## 14. Tradeoffs

**Q57. Accuracy vs. interval width — how did you navigate this for the ETA model?**
- *Strong answer:* chose an 80% interval (10th/90th percentile) as a middle ground — tighter than 95% (which would be wider and less actionable) but still practically useful; validated the choice by checking empirical coverage matched the target rather than picking the quantiles arbitrarily.

**Q58. Model complexity vs. interpretability — why not always use the most complex model?**
- *Strong answer:* LightGBM was chosen partly *because* SHAP TreeExplainer gives exact, cheap attributions — a marginal accuracy gain from a more complex/opaque model would cost the ability to explain "why did the model predict a 45-minute ETA" to stakeholders, which matters operationally (e.g. justifying a surge/delay message to customers).

**Q59. Why not build a single unified model across all forecast horizons (multi-output)?**
- *Strong answer:* considered, but a multi-output model shares capacity across targets with very different optimal feature-weightings (recent lags matter enormously for 30-min-ahead, much less for next-day) — five independently-tuned single-output models let each horizon's model specialize, at the cost of more training/maintenance overhead.

---

## 15. Failure Cases

**Q60. Where does this system fail today, concretely?**
- *Strong answer:* long-distance (10km+) ETA predictions have the highest error band; very new restaurants/customers get noisy historical features; forecasts beyond a few hours rely on seasonality alone once recent-momentum features run out.

**Q61. What input would make the demand model most wrong?**
- *Strong answer:* a genuine regime break not present in training — e.g. a new restaurant opening right next to an existing very popular one and cannibalizing its demand suddenly; the lag/rolling features would keep predicting the old pattern for a while (lag-based features are inherently reactive, not anticipatory, to structural breaks).

**Q62. What would happen if a whole zone lost power/connectivity for a day (no orders logged)?**
- *Strong answer:* the demand panel's dense grid fill (`_build_demand_panel`) would record zero-order buckets rather than missing rows, which the model would (incorrectly) treat as genuine zero demand rather than a data outage — a production system would need an explicit outage flag to avoid learning a false "demand dropped to zero" pattern.

**Q63. How would you detect the model quietly degrading in production before it's visible in aggregate MAE?**
- *Strong answer:* segment-level monitoring (by zone/peak-hour/distance-band, mirroring the `error_breakdown` utility already built here) — aggregate MAE can stay flat while a specific high-value segment silently degrades.
