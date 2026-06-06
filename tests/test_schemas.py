"""Tests for the API request/response schemas."""

import pytest
from pydantic import ValidationError

from app.schemas import PredictRequest


def _valid_payload():
    """Return a payload that satisfies every PredictRequest field."""
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
    """A well-formed payload constructs a PredictRequest with coerced enum values."""
    req = PredictRequest(**_valid_payload())
    assert req.ram_gb == 8
    assert req.cpu_brand.value == "Intel Core i5"


def test_invalid_enum_value():
    """An out-of-enum cpu_brand is rejected with a ValidationError."""
    payload = _valid_payload()
    payload["cpu_brand"] = "Pentium"
    with pytest.raises(ValidationError):
        PredictRequest(**payload)


def test_negative_inches_rejected():
    """A non-positive inches value violates the gt=0 constraint."""
    payload = _valid_payload()
    payload["inches"] = -1
    with pytest.raises(ValidationError):
        PredictRequest(**payload)
