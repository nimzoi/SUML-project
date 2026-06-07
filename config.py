"""Load and validate the project configuration from config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field


class SyntheticConfig(BaseModel):
    """Settings for synthetic data generation."""

    enabled: bool = True
    n_rows: int = Field(1000, gt=0)
    seed: int = Field(42, ge=0)


class DataConfig(BaseModel):
    """Dataset location, schema and split settings."""

    raw_path: str
    synthetic: SyntheticConfig
    target: str
    numeric_features: List[str] = Field(min_length=1)
    categorical_features: List[str] = Field(min_length=1)
    test_size: float = Field(0.2, gt=0, lt=1)
    random_state: int = 42


class ModelConfig(BaseModel):
    """AutoML and artifact settings."""

    task: Literal["regression", "classification"] = "regression"
    time_budget_s: int = Field(60, gt=0)
    metric: Literal["mae", "mse", "rmse", "r2", "mape"] = "mae"
    estimator_list: List[str] = Field(default_factory=lambda: ["lgbm", "rf", "extra_tree"])
    ensemble: bool = True
    log_target: bool = False
    monotone_increasing: List[str] = Field(default_factory=list)
    artifact_dir: str = "model/artifacts"
    artifact_name: str = "model.joblib"
    metrics_name: str = "metrics.json"
    seed: int = 42


class ValidationConfig(BaseModel):
    """Quality gates for data validation and model promotion."""

    min_rows: int = Field(100, gt=0)
    min_r2: float = Field(0.7, le=1.0)
    max_mae: Optional[float] = Field(25000.0, gt=0)


class MlflowTrackingConfig(BaseModel):
    """Optional MLflow experiment tracking settings."""

    enabled: bool = True
    tracking_uri: str = "mlruns"
    experiment_name: str = "laptop-price-automl"


class TrackingConfig(BaseModel):
    """Experiment tracking configuration."""

    mlflow: MlflowTrackingConfig = Field(default_factory=MlflowTrackingConfig)


class ApiConfig(BaseModel):
    """FastAPI host/port."""

    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1, le=65535)


class UiConfig(BaseModel):
    """Streamlit settings."""

    api_url: str = "http://localhost:8000"


class AppConfig(BaseModel):
    """Top-level application configuration."""

    data: DataConfig
    model: ModelConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    api: ApiConfig
    ui: UiConfig

    @property
    def feature_columns(self) -> List[str]:
        """All model input columns (numeric first, then categorical)."""
        return self.data.numeric_features + self.data.categorical_features

    @property
    def artifact_path(self) -> Path:
        """Full path to the persisted model artifact."""
        return Path(self.model.artifact_dir) / self.model.artifact_name

    @property
    def metrics_path(self) -> Path:
        """Full path to the persisted metrics/metadata file."""
        return Path(self.model.artifact_dir) / self.model.metrics_name


def load_config(path: Union[str, Path] = "config.yaml") -> AppConfig:
    """Read config.yaml and return a validated AppConfig.

    Raises FileNotFoundError if the file is missing and pydantic.ValidationError
    if required keys are absent or malformed.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return AppConfig(**raw)
