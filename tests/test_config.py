"""Tests for the configuration loader."""

import pytest

from config import AppConfig, load_config


def test_load_config_returns_appconfig():
    """load_config parses config.yaml into an AppConfig with the expected target/task."""
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.data.target == "Price"
    assert cfg.model.task == "regression"


def test_feature_columns_combines_numeric_and_categorical():
    """feature_columns concatenates the numeric features followed by the categorical ones."""
    cfg = load_config()
    assert cfg.feature_columns == cfg.data.numeric_features + cfg.data.categorical_features


def test_artifact_and_metrics_paths():
    """artifact_path and metrics_path resolve to the configured file names."""
    cfg = load_config()
    assert cfg.artifact_path.name == "model.joblib"
    assert cfg.metrics_path.name == "metrics.json"


def test_missing_file_raises():
    """load_config raises FileNotFoundError when the config path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist.yaml")
