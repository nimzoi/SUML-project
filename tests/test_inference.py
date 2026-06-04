"""Tests for shared inference helpers."""

import joblib

from app.inference import predict_price, to_feature_row
from config import load_config


def _payload():
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


def test_to_feature_row_shape_and_columns():
    row = to_feature_row(_payload())
    assert row.shape == (1, 13)
    assert set(row.columns) == {
        "Company",
        "TypeName",
        "Inches",
        "Ram",
        "Weight",
        "Touchscreen",
        "Ips",
        "ppi",
        "Cpu_brand",
        "SSD",
        "HDD",
        "Gpu_brand",
        "Os",
    }


def test_predict_price_positive(trained_model):
    model = joblib.load(trained_model.artifact_path)
    assert predict_price(model, _payload()) > 0
