"""FastAPI service exposing the delivery-time model."""

from __future__ import annotations

import json
import logging
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import HealthResponse, PredictRequest, PredictResponse
from config import load_config

logger = logging.getLogger(__name__)
config = load_config()

app = FastAPI(title="Food Delivery ETA API", version="1.0.0")

# Map snake_case request fields to the model's original column names.
_REQUEST_TO_COLUMN = {
    "distance_km": "Distance_km",
    "weather": "Weather",
    "traffic_level": "Traffic_Level",
    "time_of_day": "Time_of_Day",
    "vehicle_type": "Vehicle_Type",
    "preparation_time_min": "Preparation_Time_min",
    "courier_experience_yrs": "Courier_Experience_yrs",
}

_cache: Dict[str, object] = {}


def _load_model():
    """Lazily load and cache the model artifact (None if not trained yet)."""
    if "model" not in _cache and config.artifact_path.exists():
        _cache["model"] = joblib.load(config.artifact_path)
        logger.info("Loaded model from %s", config.artifact_path)
    return _cache.get("model")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe; reports whether a model is loaded."""
    return HealthResponse(status="ok", model_loaded=_load_model() is not None)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict delivery time (minutes) for a single order."""
    model = _load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available. Train it first.")
    row = {
        column: (
            getattr(request, field).value
            if hasattr(getattr(request, field), "value")
            else getattr(request, field)
        )
        for field, column in _REQUEST_TO_COLUMN.items()
    }
    prediction = float(model.predict(pd.DataFrame([row]))[0])
    return PredictResponse(eta_minutes=round(prediction, 1))


@app.get("/model-info")
def model_info() -> Dict:
    """Return the persisted metrics/metadata for the current model."""
    if not config.metrics_path.exists():
        raise HTTPException(status_code=503, detail="No model metrics found. Train first.")
    with config.metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
