"""Shared inference helpers used by both the API and the standalone UI."""

from __future__ import annotations

from typing import Dict

import pandas as pd

# Map snake_case request fields to the model's engineered column names.
REQUEST_TO_COLUMN = {
    "company": "Company",
    "type_name": "TypeName",
    "inches": "Inches",
    "ram_gb": "Ram",
    "weight_kg": "Weight",
    "touchscreen": "Touchscreen",
    "ips": "Ips",
    "ppi": "ppi",
    "cpu_brand": "Cpu_brand",
    "ssd_gb": "SSD",
    "hdd_gb": "HDD",
    "gpu_brand": "Gpu_brand",
    "os": "Os",
}


def to_feature_row(payload: Dict) -> pd.DataFrame:
    """Map a snake_case request payload to a one-row model-input DataFrame.

    Column order does not matter: the pipeline's ColumnTransformer selects by name.
    """
    row = {column: payload[field] for field, column in REQUEST_TO_COLUMN.items()}
    return pd.DataFrame([row])


def predict_price(model, payload: Dict) -> float:
    """Predict the laptop price for one configuration, rounded to 2 decimals."""
    return round(float(model.predict(to_feature_row(payload))[0]), 2)
