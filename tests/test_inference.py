"""Tests for shared inference helpers."""

import joblib

from app.inference import predict_price, to_feature_row
from app.schemas import PredictRequest


def _request():
    """Return a validated PredictRequest for a single laptop."""
    return PredictRequest(
        company="Dell",
        type_name="Notebook",
        inches=15.6,
        ram_gb=8,
        weight_kg=1.6,
        touchscreen=0,
        ips=1,
        ppi=141.2,
        cpu_brand="Intel Core i5",
        ssd_gb=256,
        hdd_gb=0,
        gpu_brand="Intel",
        os="Windows",
    )


def test_to_feature_row_shape_and_columns():
    """to_feature_row maps the request to one row with the 13 engineered model columns."""
    row = to_feature_row(_request())
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
        "Cpu_rank",
        "SSD",
        "HDD",
        "Gpu_brand",
        "Os",
    }


def test_predict_price_positive(trained_model):
    """predict_price returns a positive price for the saved pipeline."""
    model = joblib.load(trained_model.artifact_path)
    assert predict_price(model, _request()) > 0
