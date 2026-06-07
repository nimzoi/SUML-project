"""FastAPI service exposing the laptop price model."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, Optional
from uuid import uuid4

import joblib
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, status

from app.explain import explain_prediction, price_band, price_sensitivity
from app.inference import predict_price
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ContributionResponse,
    DataSchemaResponse,
    ExplainResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PriceBandResponse,
    RetrainJobStatus,
    RetrainRequest,
    SensitivityPoint,
)
from config import AppConfig, load_config
from data.contracts import RAW_TEXT_COLUMNS
from data.load import load_data
from data.monitoring import build_data_profile, compare_data_profiles
from model.retraining import retrain_with_validation, validate_training_data
from model.schemas import DataDriftReport, DataValidationReport, ModelInfo

logger = logging.getLogger(__name__)
config: AppConfig = load_config()

OPENAPI_TAGS = [
    {"name": "operations", "description": "Health checks and data validation."},
    {"name": "prediction", "description": "Online laptop price inference."},
    {"name": "model", "description": "Model metadata and retraining pipeline."},
]

app = FastAPI(
    title="Laptop Price API",
    version="1.1.0",
    description=(
        "API for laptop price prediction, data validation and controlled model retraining. "
        "Swagger UI is available at `/docs`, ReDoc at `/redoc`."
    ),
    openapi_tags=OPENAPI_TAGS,
)

_cache: Dict[str, object] = {}
_jobs: Dict[str, RetrainJobStatus] = {}
_jobs_lock = Lock()
_retrain_api_key = os.getenv("RETRAIN_API_KEY")
RAM_SENSITIVITY_VALUES = [4, 8, 12, 16, 24, 32, 64]


def _load_model():
    """Lazily load and cache the model artifact (None if not trained yet)."""
    if "model" not in _cache and config.artifact_path.exists():
        _cache["model"] = joblib.load(config.artifact_path)
        logger.info("Loaded model from %s", config.artifact_path)
    return _cache.get("model")


def _load_model_info(required: bool = True) -> Optional[ModelInfo]:
    """Load the persisted model report, optionally tolerating missing metrics."""
    if not config.metrics_path.exists():
        if required:
            raise HTTPException(status_code=503, detail="No model metrics found. Train first.")
        return None
    with config.metrics_path.open("r", encoding="utf-8") as handle:
        return ModelInfo(**json.load(handle))


def _require_retraining_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Require X-API-Key for retraining when RETRAIN_API_KEY is configured."""
    if _retrain_api_key and x_api_key != _retrain_api_key:
        raise HTTPException(status_code=401, detail="Invalid retraining API key.")


def _now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


def _has_active_job() -> bool:
    """Return True when a retraining job is already queued or running."""
    return any(job.status in {"queued", "running"} for job in _jobs.values())


def _run_retraining_job(job_id: str, request: RetrainRequest) -> None:
    """Execute one retraining job and update the in-memory status store."""
    with _jobs_lock:
        _jobs[job_id].status = "running"
        _jobs[job_id].detail = "Retraining is running."
    try:
        result = retrain_with_validation(
            config,
            time_budget_s=request.time_budget_s,
            min_r2=request.min_r2,
            max_mae=request.max_mae,
        )
    except Exception as ex:  # pylint: disable=broad-exception-caught
        logger.exception("Retraining job %s failed unexpectedly", job_id)
        with _jobs_lock:
            job = _jobs[job_id]
            job.status = "failed"
            job.detail = f"Retraining failed unexpectedly: {ex}"
            job.finished_at = _now()
        return
    with _jobs_lock:
        job = _jobs[job_id]
        job.status = result.status
        job.detail = result.detail
        job.finished_at = _now()
        job.result = result
        if result.promoted:
            _cache.pop("model", None)


@app.get("/health", response_model=HealthResponse, tags=["operations"], summary="Health check")
def health() -> HealthResponse:
    """Liveness probe; reports whether a model is loaded."""
    return HealthResponse(status="ok", model_loaded=_load_model() is not None)


@app.post("/predict", response_model=PredictResponse, tags=["prediction"], summary="Predict price")
def predict(request: PredictRequest) -> PredictResponse:
    """Predict the price for a single laptop configuration."""
    model = _load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available. Train it first.")
    return PredictResponse(price=predict_price(model, request))


@app.post(
    "/explain",
    response_model=ExplainResponse,
    tags=["prediction"],
    summary="Explain one prediction",
)
def explain(request: PredictRequest) -> ExplainResponse:
    """Predict a price and return local feature contributions plus RAM sensitivity."""
    model = _load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available. Train it first.")

    price = predict_price(model, request)
    info = _load_model_info(required=False)
    typical_error = float(info.mae) if info else price * 0.15
    band = price_band(price, typical_error)
    sensitivity = price_sensitivity(model, request, "ram_gb", RAM_SENSITIVITY_VALUES)

    return ExplainResponse(
        price=price,
        typical_error=typical_error,
        band=PriceBandResponse(low=band.low, high=band.high),
        contributions=[
            ContributionResponse(label=item.label, amount=item.amount)
            for item in explain_prediction(model, request)[:8]
        ],
        sensitivity_field="ram_gb",
        sensitivity=[
            SensitivityPoint(value=value, price=sensitivity_price)
            for value, sensitivity_price in sensitivity.items()
        ],
    )


@app.post(
    "/predict-batch",
    response_model=BatchPredictResponse,
    tags=["prediction"],
    summary="Predict prices for multiple laptops",
)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Predict prices for up to 100 laptop configurations in one request."""
    model = _load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available. Train it first.")
    return BatchPredictResponse(prices=[predict_price(model, item) for item in request.items])


@app.get("/model-info", response_model=ModelInfo, tags=["model"], summary="Get model metadata")
def model_info() -> ModelInfo:
    """Return the persisted metrics/metadata for the current model."""
    return _load_model_info()


@app.get(
    "/data-schema",
    response_model=DataSchemaResponse,
    tags=["operations"],
    summary="Get data and validation contract",
)
def data_schema() -> DataSchemaResponse:
    """Return raw CSV requirements, engineered features and validation gates."""
    return DataSchemaResponse(
        raw_required_columns=RAW_TEXT_COLUMNS + ["Inches", config.data.target],
        feature_columns=config.feature_columns,
        numeric_features=config.data.numeric_features,
        categorical_features=config.data.categorical_features,
        target=config.data.target,
        validation_gates=config.validation.model_dump(),  # pylint: disable=no-member
    )


@app.post(
    "/validate-data",
    response_model=DataValidationReport,
    tags=["operations"],
    summary="Validate the current training dataset",
)
def validate_data() -> DataValidationReport:
    """Validate the configured training data without retraining the model."""
    report = validate_training_data(config)
    if not report.ok:
        raise HTTPException(status_code=422, detail=report.model_dump())
    return report


@app.get(
    "/data-drift",
    response_model=DataDriftReport,
    tags=["operations"],
    summary="Check current training-data drift",
)
def data_drift() -> DataDriftReport:
    """Compare current training data against the profile saved with model metrics."""
    info = _load_model_info(required=False)
    try:
        current_profile = build_data_profile(load_data(config), config)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as ex:
        raise HTTPException(
            status_code=422, detail=f"Current data cannot be profiled: {ex}"
        ) from ex

    reference_profile = info.data_profile if info and info.data_profile else current_profile
    report = compare_data_profiles(reference_profile, current_profile)
    if not info or not info.data_profile:
        return report.model_copy(
            update={
                "detail": (
                    "Brak profilu referencyjnego w metrics.json; aktualne dane zostały użyte "
                    "jako baseline. Uruchom retrening, żeby zapisać profil referencyjny."
                )
            }
        )
    return report


@app.post(
    "/retrain",
    response_model=RetrainJobStatus,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["model"],
    summary="Start a staged retraining job",
)
def retrain(
    request: RetrainRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(_require_retraining_key),
) -> RetrainJobStatus:
    """Queue retraining; artifacts are promoted only after validation gates pass."""
    with _jobs_lock:
        if _has_active_job():
            raise HTTPException(status_code=409, detail="A retraining job is already active.")
        job_id = str(uuid4())
        job = RetrainJobStatus(
            job_id=job_id,
            status="queued",
            detail="Retraining has been queued.",
            started_at=_now(),
        )
        _jobs[job_id] = job
    background_tasks.add_task(_run_retraining_job, job_id, request)
    return job


@app.get(
    "/retrain/{job_id}",
    response_model=RetrainJobStatus,
    tags=["model"],
    summary="Get retraining job status",
)
def retrain_status(job_id: str) -> RetrainJobStatus:
    """Return the latest in-memory status for a retraining job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Retraining job not found.")
    return job
