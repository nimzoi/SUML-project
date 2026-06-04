"""Tests for the configuration loader."""

import pytest

from config import AppConfig, load_config


def test_load_config_returns_appconfig():
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.data.target == "Delivery_Time_min"
    assert cfg.model.task == "regression"


def test_feature_columns_combines_numeric_and_categorical():
    cfg = load_config()
    assert cfg.feature_columns == cfg.data.numeric_features + cfg.data.categorical_features


def test_artifact_and_metrics_paths():
    cfg = load_config()
    assert cfg.artifact_path.name == "model.joblib"
    assert cfg.metrics_path.name == "metrics.json"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist.yaml")
