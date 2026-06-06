"""Tests for regression metrics."""

from model.evaluate import regression_metrics
from model.schemas import RegressionMetrics


def test_perfect_prediction():
    """Identical predictions give zero MAE/RMSE and an R2 of 1.0."""
    metrics = regression_metrics([1, 2, 3], [1, 2, 3])
    assert metrics.mae == 0.0
    assert metrics.rmse == 0.0
    assert metrics.r2 == 1.0


def test_returns_regression_metrics_model():
    """regression_metrics returns a validated RegressionMetrics with float fields."""
    metrics = regression_metrics([1, 2, 3, 4], [1.5, 2.5, 2.0, 4.0])
    assert isinstance(metrics, RegressionMetrics)
    assert all(isinstance(value, float) for value in (metrics.mae, metrics.rmse, metrics.r2))
