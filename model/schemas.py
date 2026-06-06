"""Typed contracts for the model layer: evaluation metrics and the persisted report."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class RegressionMetrics(BaseModel):
    """Holdout regression scores (rounded floats)."""

    mae: float
    rmse: float
    r2: float


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
