"""Shared inference helpers used by both the API and the standalone UI."""

from __future__ import annotations

import pandas as pd

from app.schemas import PredictRequest
from data.features import CPU_RANK

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
    "ssd_gb": "SSD",
    "hdd_gb": "HDD",
    "gpu_brand": "Gpu_brand",
    "os": "Os",
}


def to_feature_row(request: PredictRequest) -> pd.DataFrame:
    """Map a validated request to a one-row model-input DataFrame.

    Enums are dumped to their string values (the form the model was trained on) and the
    CPU brand is converted to its ordinal rank. Column order does not matter — the
    pipeline's ColumnTransformer selects by name.
    """
    payload = request.model_dump(mode="json")
    row = {column: payload[field] for field, column in REQUEST_TO_COLUMN.items()}
    row["Cpu_rank"] = CPU_RANK[payload["cpu_brand"]]
    return pd.DataFrame([row])


def predict_price(model, request: PredictRequest) -> float:
    """Predict the laptop price for one configuration, rounded to 2 decimals."""
    return round(float(model.predict(to_feature_row(request))[0]), 2)
