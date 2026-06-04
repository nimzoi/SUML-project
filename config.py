"""Load and validate the project configuration from config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import yaml
from pydantic import BaseModel, Field


class SyntheticConfig(BaseModel):
    """Settings for synthetic data generation."""

    enabled: bool = True
    n_rows: int = 1000
    seed: int = 42


class DataConfig(BaseModel):
    """Dataset location, schema and split settings."""

    raw_path: str
    synthetic: SyntheticConfig
    target: str
    numeric_features: List[str]
    categorical_features: List[str]
    test_size: float = 0.2
    random_state: int = 42


class ModelConfig(BaseModel):
    """AutoML and artifact settings."""

    task: str = "regression"
    time_budget_s: int = 60
    metric: str = "mae"
    estimator_list: List[str] = Field(default_factory=lambda: ["lgbm", "rf", "extra_tree"])
    artifact_dir: str = "model/artifacts"
    artifact_name: str = "model.joblib"
    metrics_name: str = "metrics.json"
    seed: int = 42


class ApiConfig(BaseModel):
    """FastAPI host/port."""

    host: str = "0.0.0.0"
    port: int = 8000


class UiConfig(BaseModel):
    """Streamlit settings."""

    api_url: str = "http://localhost:8000"


class AppConfig(BaseModel):
    """Top-level application configuration."""

    data: DataConfig
    model: ModelConfig
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
