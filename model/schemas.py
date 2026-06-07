"""Typed contracts for model evaluation, persisted reports and retraining gates."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RegressionMetrics(BaseModel):
    """Holdout regression scores (rounded floats)."""

    mae: float
    rmse: float
    r2: float


class NumericFeatureProfile(BaseModel):
    """Compact summary of one numeric feature distribution."""

    mean: float
    std: float
    min: float
    max: float
    missing_rate: float


class CategoricalFeatureProfile(BaseModel):
    """Compact summary of one categorical feature distribution."""

    frequencies: Dict[str, float] = Field(default_factory=dict)
    missing_rate: float
    unique_count: int


class DataProfile(BaseModel):
    """Reference profile used to compare future training data against current artifacts."""

    n_rows: int
    numeric: Dict[str, NumericFeatureProfile] = Field(default_factory=dict)
    categorical: Dict[str, CategoricalFeatureProfile] = Field(default_factory=dict)


class DriftFeatureReport(BaseModel):
    """Drift score for one feature."""

    feature: str
    kind: Literal["numeric", "categorical"]
    score: float
    threshold: float
    drifted: bool
    detail: str


class DataDriftReport(BaseModel):
    """Dataset drift report comparing the saved reference profile with current data."""

    ok: bool
    reference_rows: int
    current_rows: int
    drifted_features: int
    features: List[DriftFeatureReport] = Field(default_factory=list)
    detail: str


class ModelInfo(BaseModel):
    """Everything persisted to ``metrics.json`` and served at ``GET /model-info``."""

    mae: float
    rmse: float
    r2: float
    best_estimator: str
    training_date: str
    n_rows: int
    data_source: str
    feature_columns: List[str]
    target: str
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    data_profile: Optional[DataProfile] = None
    mlflow_run_id: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None


class DataValidationReport(BaseModel):
    """Result of validating the current training dataset."""

    ok: bool
    source: str
    n_rows: int
    min_rows: int
    missing_columns: List[str] = Field(default_factory=list)
    extra_columns: List[str] = Field(default_factory=list)
    null_counts: Dict[str, int] = Field(default_factory=dict)
    detail: str


class ModelValidationReport(BaseModel):
    """Result of checking a newly trained model against promotion gates."""

    ok: bool
    min_r2: float
    max_mae: Optional[float]
    checks: Dict[str, bool]
    detail: str


class RetrainingResult(BaseModel):
    """Outcome of a staged retraining run."""

    status: Literal["succeeded", "failed"]
    promoted: bool
    detail: str
    data_validation: DataValidationReport
    model_validation: Optional[ModelValidationReport] = None
    model_info: Optional[ModelInfo] = None
