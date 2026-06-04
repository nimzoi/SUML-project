"""Shared inference helpers used by both the API and the standalone UI."""

from __future__ import annotations

from typing import Dict

import pandas as pd

# Map snake_case request fields to the model's original column names.
REQUEST_TO_COLUMN = {
    "distance_km": "Distance_km",
    "weather": "Weather",
    "traffic_level": "Traffic_Level",
    "time_of_day": "Time_of_Day",
    "vehicle_type": "Vehicle_Type",
    "preparation_time_min": "Preparation_Time_min",
    "courier_experience_yrs": "Courier_Experience_yrs",
}


def to_feature_row(payload: Dict) -> pd.DataFrame:
    """Map a snake_case request payload to a one-row model-input DataFrame.

    Column order does not matter: the pipeline's ColumnTransformer selects by name.
    """
    row = {column: payload[field] for field, column in REQUEST_TO_COLUMN.items()}
    return pd.DataFrame([row])


def predict_minutes(model, payload: Dict) -> float:
    """Predict delivery time (minutes) for one order, rounded to 1 decimal."""
    return round(float(model.predict(to_feature_row(payload))[0]), 1)
