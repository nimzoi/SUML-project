"""Tests for the API request/response schemas."""

import pytest
from pydantic import ValidationError

from app.schemas import PredictRequest


def _valid_payload():
    return {
        "company": "Dell",
        "type_name": "Notebook",
        "inches": 15.6,
        "ram_gb": 8,
        "weight_kg": 1.6,
        "touchscreen": 0,
        "ips": 1,
        "ppi": 141.2,
        "cpu_brand": "Intel Core i5",
        "ssd_gb": 256,
        "hdd_gb": 0,
        "gpu_brand": "Intel",
        "os": "Windows",
    }


def test_valid_request():
    req = PredictRequest(**_valid_payload())
    assert req.ram_gb == 8
    assert req.cpu_brand.value == "Intel Core i5"


def test_invalid_enum_value():
    payload = _valid_payload()
    payload["cpu_brand"] = "Pentium"
    with pytest.raises(ValidationError):
        PredictRequest(**payload)


def test_negative_inches_rejected():
    payload = _valid_payload()
    payload["inches"] = -1
    with pytest.raises(ValidationError):
        PredictRequest(**payload)
