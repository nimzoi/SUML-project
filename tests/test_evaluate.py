"""Tests for regression metrics."""

from model.evaluate import regression_metrics


def test_perfect_prediction():
    """Identical predictions give zero MAE/RMSE and an R2 of 1.0."""
    metrics = regression_metrics([1, 2, 3], [1, 2, 3])
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_keys_present_and_floats():
    """regression_metrics returns exactly mae/rmse/r2, all as floats."""
    metrics = regression_metrics([1, 2, 3, 4], [1.5, 2.5, 2.0, 4.0])
    assert set(metrics) == {"mae", "rmse", "r2"}
    assert all(isinstance(v, float) for v in metrics.values())
