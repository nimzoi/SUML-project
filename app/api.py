"""FastAPI service exposing the laptop price model."""

from __future__ import annotations

import json
import logging
from typing import Dict

import joblib
from fastapi import FastAPI, HTTPException

from app.inference import predict_price
from app.schemas import HealthResponse, PredictRequest, PredictResponse
from config import load_config
from model.schemas import ModelInfo

logger = logging.getLogger(__name__)
config = load_config()

app = FastAPI(title="Laptop Price API", version="1.0.0")

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
    """Predict the price for a single laptop configuration."""
    model = _load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available. Train it first.")
    return PredictResponse(price=predict_price(model, request))


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    """Return the persisted metrics/metadata for the current model."""
    if not config.metrics_path.exists():
        raise HTTPException(status_code=503, detail="No model metrics found. Train first.")
    with config.metrics_path.open("r", encoding="utf-8") as handle:
        return ModelInfo(**json.load(handle))
