"""Tests for the synthetic dataset generator -- schema/edge-case checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.generate_dataset import _make_zones, _make_restaurants, _make_weather_traffic


def test_zones_schema_and_ranges():
    rng = np.random.default_rng(0)
    zones = _make_zones(rng, n_zones=5)
    assert len(zones) == 5
    assert zones["zone_id"].is_unique
    assert (zones["demand_multiplier"] > 0).all()
    assert (zones["base_traffic_speed_kmph"] > 0).all()


def test_restaurants_reference_valid_zones():
    rng = np.random.default_rng(0)
    zones = _make_zones(rng, n_zones=5)
    restaurants = _make_restaurants(rng, n_restaurants=20, zones=zones)
    assert restaurants["restaurant_id"].is_unique
    assert set(restaurants["zone_id"]).issubset(set(zones["zone_id"]))
    assert (restaurants["base_prep_time_min"] > 0).all()
    assert (restaurants["rating"].between(1, 5)).all()


def test_weather_traffic_covers_every_zone_and_hour():
    rng = np.random.default_rng(0)
    zones = _make_zones(rng, n_zones=3)
    n_days = 2
    wt = _make_weather_traffic(rng, n_days=n_days, start_date="2026-01-01", zones=zones)
    expected_rows = 3 * n_days * 24
    assert len(wt) == expected_rows
    assert (wt["traffic_multiplier"] > 0).all()
    assert wt["weather_condition"].isin(["Clear", "Light Rain", "Heavy Rain"]).all()


def test_weather_traffic_no_negative_rain():
    rng = np.random.default_rng(1)
    zones = _make_zones(rng, n_zones=2)
    wt = _make_weather_traffic(rng, n_days=5, start_date="2026-01-01", zones=zones)
    assert (wt["rain_mm"] >= 0).all()
