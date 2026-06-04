"""Tests for the API request/response schemas."""

import pytest
from pydantic import ValidationError

from app.schemas import PredictRequest


def _valid_payload():
    return {
        "distance_km": 7.9,
        "weather": "Clear",
        "traffic_level": "Medium",
        "time_of_day": "Afternoon",
        "vehicle_type": "Scooter",
        "preparation_time_min": 12,
        "courier_experience_yrs": 2.0,
    }


def test_valid_request():
    req = PredictRequest(**_valid_payload())
    assert req.distance_km == 7.9
    assert req.weather.value == "Clear"


def test_invalid_enum_value():
    payload = _valid_payload()
    payload["weather"] = "Sunny"
    with pytest.raises(ValidationError):
        PredictRequest(**payload)


def test_negative_distance_rejected():
    payload = _valid_payload()
    payload["distance_km"] = -1
    with pytest.raises(ValidationError):
        PredictRequest(**payload)
