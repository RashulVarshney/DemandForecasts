-- SQL feature-extraction queries.
--
-- These mirror (in pure SQL) a subset of what src/features/build_features.py
-- computes in pandas, to demonstrate that the same leakage-safe lag/rolling
-- logic can be expressed with window functions -- useful when features need
-- to be computed inside a warehouse (e.g. a dbt model or a Snowflake/BigQuery
-- feature pipeline) rather than in a Python service.
--
-- All window functions here use `ORDER BY bucket_ts ROWS BETWEEN n PRECEDING
-- AND 1 PRECEDING` (or LAG with offset >= 1) so that, exactly like in the
-- pandas pipeline, the current bucket's own order_count can NEVER appear in
-- its own feature -- this is the SQL equivalent of "shift before rolling".

-- =====================================================================
-- 1. Repeat customers: how many distinct orders per customer, and what
--    share of total orders come from customers who ordered more than once.
-- =====================================================================
WITH customer_orders AS (
    SELECT customer_id, COUNT(*) AS order_count
    FROM read_parquet('data/raw/orders.parquet')
    WHERE NOT is_cancelled
    GROUP BY customer_id
)
SELECT
    CASE WHEN order_count = 1 THEN 'one_time' ELSE 'repeat' END AS customer_type,
    COUNT(*) AS n_customers,
    SUM(order_count) AS total_orders,
    ROUND(100.0 * SUM(order_count) / SUM(SUM(order_count)) OVER (), 2) AS pct_of_orders
FROM customer_orders
GROUP BY customer_type;

-- =====================================================================
-- 2. 30-min bucketed demand panel (restaurant x zone x bucket), the base
--    grain the forecasting model is trained on.
-- =====================================================================
CREATE OR REPLACE TEMP VIEW demand_buckets AS
SELECT
    restaurant_id,
    zone_id,
    time_bucket(INTERVAL '30 minutes', order_timestamp) AS bucket_ts,
    COUNT(*) AS order_count,
    AVG(basket_value_inr) AS avg_basket_value
FROM read_parquet('data/raw/orders.parquet')
WHERE NOT is_cancelled
GROUP BY restaurant_id, zone_id, bucket_ts;

SELECT * FROM demand_buckets ORDER BY restaurant_id, bucket_ts LIMIT 20;

-- =====================================================================
-- 3. Lag features via window functions (previous bucket, previous day
--    same time, previous week same time) per restaurant.
--    NOTE: this assumes a fully dense bucket grid (no missing buckets);
--    the production pandas pipeline fills gaps explicitly before lagging
--    to avoid silently lagging across a missing bucket.
-- =====================================================================
SELECT
    restaurant_id,
    bucket_ts,
    order_count,
    LAG(order_count, 1) OVER (PARTITION BY restaurant_id ORDER BY bucket_ts) AS lag_30m,
    LAG(order_count, 2) OVER (PARTITION BY restaurant_id ORDER BY bucket_ts) AS lag_1h,
    LAG(order_count, 48) OVER (PARTITION BY restaurant_id ORDER BY bucket_ts) AS lag_1d,
    LAG(order_count, 336) OVER (PARTITION BY restaurant_id ORDER BY bucket_ts) AS lag_7d
FROM demand_buckets
ORDER BY restaurant_id, bucket_ts;

-- =====================================================================
-- 4. Rolling mean/std over the PAST 8 hours (16 buckets), excluding the
--    current bucket -- window frame is "16 PRECEDING to 1 PRECEDING".
-- =====================================================================
SELECT
    restaurant_id,
    bucket_ts,
    order_count,
    AVG(order_count) OVER (
        PARTITION BY restaurant_id ORDER BY bucket_ts
        ROWS BETWEEN 16 PRECEDING AND 1 PRECEDING
    ) AS roll_mean_8h,
    STDDEV(order_count) OVER (
        PARTITION BY restaurant_id ORDER BY bucket_ts
        ROWS BETWEEN 16 PRECEDING AND 1 PRECEDING
    ) AS roll_std_8h
FROM demand_buckets
ORDER BY restaurant_id, bucket_ts;

-- =====================================================================
-- 5. Restaurant historical (expanding, past-only) average demand --
--    SQL equivalent of pandas .expanding().mean() on the shifted series.
-- =====================================================================
SELECT
    restaurant_id,
    bucket_ts,
    order_count,
    AVG(order_count) OVER (
        PARTITION BY restaurant_id ORDER BY bucket_ts
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS restaurant_hist_avg_demand
FROM demand_buckets
ORDER BY restaurant_id, bucket_ts;

-- =====================================================================
-- 6. Restaurant historical prep-time / ETA behavior for the ETA model,
--    computed the same "past orders only" way.
-- =====================================================================
SELECT
    order_id,
    restaurant_id,
    order_timestamp,
    prep_time_min,
    eta_minutes,
    AVG(prep_time_min) OVER (
        PARTITION BY restaurant_id ORDER BY order_timestamp
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS restaurant_hist_avg_prep_time,
    AVG(eta_minutes) OVER (
        PARTITION BY restaurant_id ORDER BY order_timestamp
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS restaurant_hist_avg_eta
FROM read_parquet('data/raw/orders.parquet')
WHERE NOT is_cancelled
ORDER BY restaurant_id, order_timestamp;

-- =====================================================================
-- 7. Zone-level demand at the PREVIOUS bucket (cross-restaurant signal),
--    a feature capturing "is the whole zone busy right now".
-- =====================================================================
WITH zone_buckets AS (
    SELECT
        zone_id,
        time_bucket(INTERVAL '30 minutes', order_timestamp) AS bucket_ts,
        COUNT(*) AS zone_order_count
    FROM read_parquet('data/raw/orders.parquet')
    WHERE NOT is_cancelled
    GROUP BY zone_id, bucket_ts
)
SELECT
    zone_id,
    bucket_ts,
    zone_order_count,
    LAG(zone_order_count, 1) OVER (PARTITION BY zone_id ORDER BY bucket_ts) AS zone_demand_prev_bucket
FROM zone_buckets
ORDER BY zone_id, bucket_ts;
