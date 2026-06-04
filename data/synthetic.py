"""Generate a synthetic Food Delivery dataset matching the real Kaggle schema."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

COLUMNS: List[str] = [
    "Order_ID",
    "Distance_km",
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
    "Delivery_Time_min",
]

WEATHER = ["Clear", "Rainy", "Snowy", "Foggy", "Windy"]
TRAFFIC = ["Low", "Medium", "High"]
TIME_OF_DAY = ["Morning", "Afternoon", "Evening", "Night"]
VEHICLE = ["Bike", "Scooter", "Car"]

_WEATHER_PENALTY = {"Clear": 0, "Windy": 3, "Foggy": 6, "Rainy": 8, "Snowy": 12}
_TRAFFIC_PENALTY = {"Low": 0, "Medium": 6, "High": 14}
_VEHICLE_SPEED = {"Bike": 2.5, "Scooter": 2.0, "Car": 1.6}
_NULL_COLUMNS = ["Weather", "Traffic_Level", "Time_of_Day", "Courier_Experience_yrs"]


def generate(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame with the same columns/types as the real dataset.

    Delivery time grows with distance, prep time, traffic and bad weather, and
    decreases slightly with courier experience, plus Gaussian noise. About 3% of
    cells in the same columns as the real data are set to NaN so the imputation
    path is exercised even without the real CSV.
    """
    rng = np.random.default_rng(seed)
    distance = np.round(rng.uniform(0.5, 20.0, n_rows), 2)
    prep = rng.integers(5, 30, n_rows)
    experience = np.round(rng.uniform(0.0, 10.0, n_rows), 1)
    weather = rng.choice(WEATHER, n_rows)
    traffic = rng.choice(TRAFFIC, n_rows)
    time_of_day = rng.choice(TIME_OF_DAY, n_rows)
    vehicle = rng.choice(VEHICLE, n_rows)

    minutes = (
        10.0
        + distance * np.array([_VEHICLE_SPEED[v] for v in vehicle])
        + prep * 0.8
        + np.array([_TRAFFIC_PENALTY[t] for t in traffic])
        + np.array([_WEATHER_PENALTY[w] for w in weather])
        - experience * 0.5
        + rng.normal(0.0, 5.0, n_rows)
    )
    minutes = np.clip(np.round(minutes), 5, None).astype(int)

    df = pd.DataFrame(
        {
            "Order_ID": np.arange(1, n_rows + 1),
            "Distance_km": distance,
            "Weather": weather,
            "Traffic_Level": traffic,
            "Time_of_Day": time_of_day,
            "Vehicle_Type": vehicle,
            "Preparation_Time_min": prep,
            "Courier_Experience_yrs": experience,
            "Delivery_Time_min": minutes,
        }
    )
    _inject_nulls(df, rng, frac=0.03)
    return df


def _inject_nulls(df: pd.DataFrame, rng: np.random.Generator, frac: float) -> None:
    """Set a fraction of cells to NaN in the columns that are nullable in real data."""
    n_null = int(len(df) * frac)
    for column in _NULL_COLUMNS:
        idx = rng.choice(len(df), size=n_null, replace=False)
        df.loc[idx, column] = np.nan
