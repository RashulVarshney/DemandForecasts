-- Exploratory SQL analysis over the synthetic food-delivery dataset.
--
-- Run with DuckDB directly against the Parquet files (no separate DB load
-- step needed -- this mirrors querying a data lake / lakehouse table):
--
--   duckdb -c ".read sql/exploratory_queries.sql"
--
-- or programmatically via src/utils / notebooks using duckdb.sql(...).
--
-- Convention: 'orders' below refers to read_parquet('data/raw/orders.parquet'),
-- joined with restaurants/zones/weather_traffic as needed. Cancelled orders
-- are excluded from demand/ETA analysis unless explicitly analyzing cancellations.

-- =====================================================================
-- 1. Top restaurants by completed order volume
-- =====================================================================
SELECT
    r.restaurant_id,
    r.restaurant_name,
    r.cuisine,
    COUNT(*) AS completed_orders,
    ROUND(AVG(o.eta_minutes), 1) AS avg_eta_minutes,
    ROUND(AVG(o.basket_value_inr), 1) AS avg_basket_value
FROM read_parquet('data/raw/orders.parquet') o
JOIN read_parquet('data/raw/restaurants.parquet') r USING (restaurant_id)
WHERE NOT o.is_cancelled
GROUP BY r.restaurant_id, r.restaurant_name, r.cuisine
ORDER BY completed_orders DESC
LIMIT 20;

-- =====================================================================
-- 2. Hourly order volume (platform-wide) -- reveals lunch/dinner peaks
-- =====================================================================
SELECT
    EXTRACT(HOUR FROM order_timestamp) AS hour_of_day,
    COUNT(*) AS orders,
    ROUND(AVG(eta_minutes), 1) AS avg_eta_minutes
FROM read_parquet('data/raw/orders.parquet')
WHERE NOT is_cancelled
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- =====================================================================
-- 3. Average delivery time (ETA) by zone
-- =====================================================================
SELECT
    z.zone_id,
    z.zone_name,
    COUNT(*) AS orders,
    ROUND(AVG(o.eta_minutes), 1) AS avg_eta_minutes,
    ROUND(APPROX_QUANTILE(o.eta_minutes, 0.9), 1) AS p90_eta_minutes
FROM read_parquet('data/raw/orders.parquet') o
JOIN read_parquet('data/raw/zones.parquet') z USING (zone_id)
WHERE NOT o.is_cancelled
GROUP BY z.zone_id, z.zone_name
ORDER BY avg_eta_minutes DESC;

-- =====================================================================
-- 4. Average ETA during peak vs. non-peak hours
-- =====================================================================
SELECT
    CASE
        WHEN EXTRACT(HOUR FROM order_timestamp) BETWEEN 12 AND 14 THEN 'lunch_peak'
        WHEN EXTRACT(HOUR FROM order_timestamp) BETWEEN 19 AND 21 THEN 'dinner_peak'
        ELSE 'off_peak'
    END AS period,
    COUNT(*) AS orders,
    ROUND(AVG(eta_minutes), 1) AS avg_eta_minutes,
    ROUND(APPROX_QUANTILE(eta_minutes, 0.9), 1) AS p90_eta_minutes,
    ROUND(APPROX_QUANTILE(eta_minutes, 0.95), 1) AS p95_eta_minutes
FROM read_parquet('data/raw/orders.parquet')
WHERE NOT is_cancelled
GROUP BY period
ORDER BY avg_eta_minutes DESC;

-- =====================================================================
-- 5. Restaurants with the highest cancellation rate (min 200 orders)
-- =====================================================================
SELECT
    r.restaurant_id,
    r.restaurant_name,
    COUNT(*) AS total_orders,
    ROUND(100.0 * SUM(CASE WHEN o.is_cancelled THEN 1 ELSE 0 END) / COUNT(*), 2) AS cancellation_rate_pct
FROM read_parquet('data/raw/orders.parquet') o
JOIN read_parquet('data/raw/restaurants.parquet') r USING (restaurant_id)
GROUP BY r.restaurant_id, r.restaurant_name
HAVING COUNT(*) >= 200
ORDER BY cancellation_rate_pct DESC
LIMIT 20;

-- =====================================================================
-- 6. Demand growth over time (daily order volume + 7-day moving average)
-- =====================================================================
WITH daily AS (
    SELECT
        CAST(order_timestamp AS DATE) AS order_date,
        COUNT(*) AS orders
    FROM read_parquet('data/raw/orders.parquet')
    WHERE NOT is_cancelled
    GROUP BY order_date
)
SELECT
    order_date,
    orders,
    ROUND(AVG(orders) OVER (
        ORDER BY order_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 1) AS orders_7d_moving_avg
FROM daily
ORDER BY order_date;

-- =====================================================================
-- 7. Delivery distance distribution (deciles)
-- =====================================================================
WITH decile_assigned AS (
    SELECT
        distance_km,
        eta_minutes,
        NTILE(10) OVER (ORDER BY distance_km) AS distance_decile
    FROM read_parquet('data/raw/orders.parquet')
    WHERE NOT is_cancelled
)
SELECT
    distance_decile,
    ROUND(MIN(distance_km), 2) AS min_km,
    ROUND(MAX(distance_km), 2) AS max_km,
    ROUND(AVG(eta_minutes), 1) AS avg_eta_minutes
FROM decile_assigned
GROUP BY distance_decile
ORDER BY distance_decile;

-- =====================================================================
-- 8. Demand by weekday x hour (heatmap source table)
-- =====================================================================
SELECT
    EXTRACT(DOW FROM order_timestamp) AS day_of_week,   -- 0=Sunday
    EXTRACT(HOUR FROM order_timestamp) AS hour_of_day,
    COUNT(*) AS orders
FROM read_parquet('data/raw/orders.parquet')
WHERE NOT is_cancelled
GROUP BY day_of_week, hour_of_day
ORDER BY day_of_week, hour_of_day;

-- =====================================================================
-- 9. Weather impact on ETA (joins order-level data to hourly weather)
-- =====================================================================
SELECT
    wt.weather_condition,
    COUNT(*) AS orders,
    ROUND(AVG(o.eta_minutes), 1) AS avg_eta_minutes,
    ROUND(AVG(o.travel_time_min), 1) AS avg_travel_time_min
FROM read_parquet('data/raw/orders.parquet') o
JOIN read_parquet('data/raw/weather_traffic.parquet') wt
    ON o.zone_id = wt.zone_id
    AND DATE_TRUNC('hour', o.order_timestamp) = wt.timestamp_hour
WHERE NOT o.is_cancelled
GROUP BY wt.weather_condition
ORDER BY avg_eta_minutes DESC;

-- =====================================================================
-- 10. Zone-level demand concentration -- what share of orders comes from
--     the top 3 zones? (useful for capacity-planning prioritization)
-- =====================================================================
WITH zone_totals AS (
    SELECT zone_id, COUNT(*) AS orders
    FROM read_parquet('data/raw/orders.parquet')
    WHERE NOT is_cancelled
    GROUP BY zone_id
),
ranked AS (
    SELECT *, RANK() OVER (ORDER BY orders DESC) AS rnk,
           SUM(orders) OVER () AS total_orders
    FROM zone_totals
)
SELECT
    zone_id,
    orders,
    ROUND(100.0 * orders / total_orders, 2) AS pct_of_total,
    ROUND(100.0 * SUM(orders) OVER (ORDER BY orders DESC) / total_orders, 2) AS cumulative_pct
FROM ranked
ORDER BY orders DESC;
