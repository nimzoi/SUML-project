"""Regression metrics for model evaluation."""

from __future__ import annotations

from typing import Sequence

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from model.schemas import RegressionMetrics


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> RegressionMetrics:
    """Return MAE, RMSE and R2 as a validated RegressionMetrics (rounded floats)."""
    return RegressionMetrics(
        mae=round(float(mean_absolute_error(y_true, y_pred)), 4),
        rmse=round(float(root_mean_squared_error(y_true, y_pred)), 4),
        r2=round(float(r2_score(y_true, y_pred)), 4),
    )
