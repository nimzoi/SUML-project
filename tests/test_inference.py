"""Tests for shared inference helpers."""

import joblib

from app.inference import predict_minutes, to_feature_row
from config import load_config


def _payload():
    return {
        "distance_km": 7.9,
        "weather": "Clear",
        "traffic_level": "Medium",
        "time_of_day": "Afternoon",
        "vehicle_type": "Scooter",
        "preparation_time_min": 12,
        "courier_experience_yrs": 2.0,
    }


def test_to_feature_row_shape_and_columns():
    row = to_feature_row(_payload())
    assert row.shape == (1, 7)
    assert set(row.columns) == {
        "Distance_km",
        "Weather",
        "Traffic_Level",
        "Time_of_Day",
        "Vehicle_Type",
        "Preparation_Time_min",
        "Courier_Experience_yrs",
    }


def test_predict_minutes_positive(trained_model):
    model = joblib.load(trained_model.artifact_path)
    assert predict_minutes(model, _payload()) > 0
