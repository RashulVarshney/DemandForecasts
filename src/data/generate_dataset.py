"""
Synthetic Swiggy-like food-delivery dataset generator.

IMPORTANT: This produces SYNTHETIC data. No field here is real Swiggy data.
It is designed to have realistic statistical structure (seasonality, peaks,
distance/traffic-driven ETA, restaurant heterogeneity) so that downstream
feature engineering / modeling work is non-trivial and defensible, while
being fully reproducible from a fixed seed. See data/README.md for the
full rationale.

Design summary
---------------
- `zones`: 12 delivery zones with a base demand multiplier and a base
  traffic-speed factor (denser zones = slower).
- `restaurants`: 150 restaurants distributed across zones/cuisines, each with
  its own popularity, base prep time and prep-time volatility.
- `weather_traffic`: hourly weather condition + traffic multiplier per zone,
  generated with day-level persistence (weather doesn't flip every hour) and
  a monsoon-season block with elevated rain probability.
- `orders`: an order-level fact table produced by sampling a non-homogeneous
  Poisson process per (restaurant, 5-min bucket) whose rate depends on
  hour-of-day, day-of-week, restaurant popularity, weather and a slow trend.
  Each order gets item count/basket value, a haversine-ish distance from a
  restaurant's zone centroid, and a structurally-generated ETA
  (prep time + travel time + queue/traffic noise), so ETA has real signal for
  ML models to recover.

Run:
    python -m src.data.generate_dataset --out data/raw --seed 42
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

from src.utils.config import load_config

CUISINES = ["North Indian", "South Indian", "Chinese", "Biryani", "Fast Food", "Desserts"]


def _make_zones(rng: np.random.Generator, n_zones: int) -> pd.DataFrame:
    zones = []
    for z in range(n_zones):
        # Denser / commercial zones have higher demand but slower traffic.
        density = rng.beta(2, 2)
        zones.append(
            {
                "zone_id": z,
                "zone_name": f"Zone_{z:02d}",
                "demand_multiplier": round(0.5 + 1.5 * density, 3),
                "base_traffic_speed_kmph": round(28 - 14 * density + rng.normal(0, 1.5), 2),
                "centroid_x_km": round(rng.uniform(0, 20), 2),
                "centroid_y_km": round(rng.uniform(0, 20), 2),
            }
        )
    return pd.DataFrame(zones)


def _make_restaurants(rng: np.random.Generator, n_restaurants: int, zones: pd.DataFrame) -> pd.DataFrame:
    rows = []
    zone_ids = zones["zone_id"].values
    for r in range(n_restaurants):
        zone_id = rng.choice(zone_ids)
        popularity = rng.lognormal(mean=0.0, sigma=0.6)  # long-tailed popularity
        rows.append(
            {
                "restaurant_id": r,
                "restaurant_name": f"Restaurant_{r:04d}",
                "zone_id": int(zone_id),
                "cuisine": rng.choice(CUISINES),
                "popularity_score": round(float(popularity), 3),
                "base_prep_time_min": round(float(np.clip(rng.normal(18, 6), 6, 45)), 1),
                "prep_time_volatility": round(float(np.clip(rng.normal(4, 1.5), 1, 12)), 2),
                "rating": round(float(np.clip(rng.normal(4.1, 0.35), 2.5, 5.0)), 2),
            }
        )
    return pd.DataFrame(rows)


def _make_weather_traffic(rng: np.random.Generator, n_days: int, start_date: str, zones: pd.DataFrame) -> pd.DataFrame:
    """Hourly weather/traffic per zone with day-level persistence."""
    hours = pd.date_range(start=start_date, periods=n_days * 24, freq="h")
    rows = []
    for zone_id in zones["zone_id"]:
        # Day-level rain regime (monsoon block in the middle third of the range)
        n_hours = len(hours)
        day_idx = np.arange(n_hours) // 24
        n_unique_days = day_idx.max() + 1
        monsoon_day = (n_unique_days // 3 <= np.arange(n_unique_days)) & (
            np.arange(n_unique_days) < 2 * n_unique_days // 3
        )
        rain_prob_by_day = np.where(monsoon_day, 0.45, 0.12)
        is_rainy_day = rng.random(n_unique_days) < rain_prob_by_day
        is_rainy_hour_base = is_rainy_day[day_idx]

        rain_intensity = np.where(
            is_rainy_hour_base, rng.gamma(2.0, 0.5, size=n_hours), 0.0
        )
        # Rain worsens traffic; independent random traffic congestion by hour-of-day (peak hours)
        hour_of_day = hours.hour
        peak_congestion = np.select(
            [
                np.isin(hour_of_day, [8, 9, 13, 18, 19, 20, 21]),
                np.isin(hour_of_day, [0, 1, 2, 3, 4, 5]),
            ],
            [rng.uniform(1.3, 1.9, n_hours), rng.uniform(0.6, 0.8, n_hours)],
            default=rng.uniform(0.9, 1.2, n_hours),
        )
        traffic_multiplier = np.clip(peak_congestion + 0.35 * rain_intensity, 0.5, 3.0)

        weather_condition = np.where(
            rain_intensity > 1.2, "Heavy Rain",
            np.where(rain_intensity > 0.1, "Light Rain", "Clear"),
        )

        zone_df = pd.DataFrame(
            {
                "zone_id": zone_id,
                "timestamp_hour": hours,
                "rain_mm": np.round(rain_intensity * 4, 2),
                "weather_condition": weather_condition,
                "traffic_multiplier": np.round(traffic_multiplier, 3),
            }
        )
        rows.append(zone_df)
    return pd.concat(rows, ignore_index=True)


def _demand_rate(hour: np.ndarray, dow: np.ndarray, day_offset: np.ndarray) -> np.ndarray:
    """Base per-restaurant expected orders per 5-min bucket, before zone/restaurant scaling."""
    # Lunch peak ~13:00, dinner peak ~20:00, small breakfast bump ~9:00
    lunch = 1.0 * np.exp(-((hour - 13.2) ** 2) / (2 * 1.3 ** 2))
    dinner = 1.35 * np.exp(-((hour - 20.0) ** 2) / (2 * 1.6 ** 2))
    breakfast = 0.35 * np.exp(-((hour - 9.0) ** 2) / (2 * 1.0 ** 2))
    late_night = 0.25 * np.exp(-((hour - 23.5) ** 2) / (2 * 1.2 ** 2))
    base = 0.05 + lunch + dinner + breakfast + late_night

    weekend_boost = np.where(dow >= 5, 1.25, 1.0)  # Sat/Sun
    # Slow upward trend over the 90 days (platform growth) ~ +25% over full range
    trend = 1.0 + 0.0028 * day_offset
    return base * weekend_boost * trend


def _generate_orders(
    rng: np.random.Generator,
    n_days: int,
    start_date: str,
    minute_granularity: int,
    zones: pd.DataFrame,
    restaurants: pd.DataFrame,
    weather_traffic: pd.DataFrame,
) -> pd.DataFrame:
    buckets = pd.date_range(start=start_date, periods=n_days * 24 * 60 // minute_granularity, freq=f"{minute_granularity}min")
    hour = buckets.hour.values
    dow = buckets.dayofweek.values
    day_offset = (buckets - buckets[0]).days
    base_rate_per_bucket = _demand_rate(hour, dow, day_offset) * (minute_granularity / 5.0)

    wt_lookup = weather_traffic.set_index(["zone_id", "timestamp_hour"])["traffic_multiplier"].to_dict()
    weather_lookup = weather_traffic.set_index(["zone_id", "timestamp_hour"])["weather_condition"].to_dict()

    zone_info = zones.set_index("zone_id")
    order_rows = []
    order_id = 0

    # A pool of customers per zone with power-law (Zipf-like) repeat-order behavior:
    # a small fraction of customers order far more frequently than the rest, which is
    # what makes "repeat customer" SQL analysis meaningful instead of everyone being unique.
    customers_per_zone = 2500
    customer_weights = {}
    customer_ids_by_zone = {}
    next_customer_id = 0
    for zone_id in zone_info.index:
        ids = np.arange(next_customer_id, next_customer_id + customers_per_zone)
        next_customer_id += customers_per_zone
        ranks = np.arange(1, customers_per_zone + 1)
        weights = 1.0 / ranks  # Zipf-like: top customers order much more often
        weights = weights / weights.sum()
        customer_ids_by_zone[zone_id] = ids
        customer_weights[zone_id] = weights

    for _, rest in restaurants.iterrows():
        zone = zone_info.loc[rest["zone_id"]]
        zone_mult = zone["demand_multiplier"]
        pop = rest["popularity_score"]
        rate = base_rate_per_bucket * zone_mult * pop
        # Poisson draw per bucket
        counts = rng.poisson(rate)
        nonzero_idx = np.nonzero(counts)[0]
        for idx in nonzero_idx:
            n_orders = counts[idx]
            bucket_time = buckets[idx]
            hour_ts = bucket_time.floor("h")
            traffic_mult = wt_lookup.get((rest["zone_id"], hour_ts), 1.0)
            weather = weather_lookup.get((rest["zone_id"], hour_ts), "Clear")
            zone_customer_ids = customer_ids_by_zone[rest["zone_id"]]
            zone_customer_weights = customer_weights[rest["zone_id"]]
            order_customer_ids = rng.choice(
                zone_customer_ids, size=n_orders, p=zone_customer_weights
            )
            for order_i in range(n_orders):
                offset_sec = rng.integers(0, minute_granularity * 60)
                order_time = bucket_time + pd.Timedelta(seconds=int(offset_sec))

                n_items = int(np.clip(rng.poisson(2.2) + 1, 1, 12))
                item_unit_price = rng.gamma(3.0, 60)
                basket_value = round(float(n_items * item_unit_price * rng.uniform(0.85, 1.15)), 2)

                distance_km = float(np.clip(rng.gamma(2.0, 1.4), 0.3, 15))

                prep_time = float(
                    np.clip(
                        rng.normal(rest["base_prep_time_min"], rest["prep_time_volatility"])
                        + 0.4 * max(0, n_items - 2),  # bigger orders take a bit longer to prep
                        4, 60,
                    )
                )

                effective_speed = max(6.0, zone["base_traffic_speed_kmph"] / traffic_mult)
                travel_time = distance_km / effective_speed * 60.0  # minutes

                rider_assignment_delay = float(np.clip(rng.exponential(2.5), 0, 20))
                queue_noise = float(np.clip(rng.normal(0, 3.0), -8, 15))

                eta_minutes = float(
                    np.clip(prep_time + travel_time + rider_assignment_delay + queue_noise, 8, 120)
                )

                is_cancelled = rng.random() < np.clip(0.015 + 0.01 * (eta_minutes > 60), 0, 0.2)

                order_rows.append(
                    (
                        order_id,
                        int(order_customer_ids[order_i]),
                        order_time,
                        int(rest["restaurant_id"]),
                        int(rest["zone_id"]),
                        n_items,
                        basket_value,
                        round(distance_km, 3),
                        round(prep_time, 2),
                        round(travel_time, 2),
                        round(rider_assignment_delay, 2),
                        round(eta_minutes, 2),
                        weather,
                        round(traffic_mult, 3),
                        bool(is_cancelled),
                    )
                )
                order_id += 1

    cols = [
        "order_id", "customer_id", "order_timestamp", "restaurant_id", "zone_id", "n_items", "basket_value_inr",
        "distance_km", "prep_time_min", "travel_time_min", "rider_assignment_delay_min",
        "eta_minutes", "weather_condition", "traffic_multiplier", "is_cancelled",
    ]
    orders = pd.DataFrame(order_rows, columns=cols)
    return orders.sort_values("order_timestamp").reset_index(drop=True)


def generate(out_dir: pathlib.Path, seed: int, config: dict) -> None:
    rng = np.random.default_rng(seed)
    sim = config["simulation"]

    zones = _make_zones(rng, sim["n_zones"])
    restaurants = _make_restaurants(rng, sim["n_restaurants"], zones)
    weather_traffic = _make_weather_traffic(rng, sim["n_days"], sim["start_date"], zones)
    orders = _generate_orders(
        rng, sim["n_days"], sim["start_date"], sim["minute_granularity"],
        zones, restaurants, weather_traffic,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    zones.to_parquet(out_dir / "zones.parquet", index=False)
    restaurants.to_parquet(out_dir / "restaurants.parquet", index=False)
    weather_traffic.to_parquet(out_dir / "weather_traffic.parquet", index=False)
    orders.to_parquet(out_dir / "orders.parquet", index=False)

    print(f"Generated {len(orders):,} orders across {sim['n_restaurants']} restaurants, "
          f"{sim['n_zones']} zones, {sim['n_days']} days.")
    print(f"Written to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    generate(pathlib.Path(args.out), args.seed, config)


if __name__ == "__main__":
    main()
