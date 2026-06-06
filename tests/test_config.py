"""Tests for the configuration loader."""

import pytest
from pydantic import ValidationError

from config import AppConfig, DataConfig, ModelConfig, SyntheticConfig, load_config


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


def test_rejects_invalid_test_size():
    """test_size outside (0, 1) is rejected by the config validators."""
    with pytest.raises(ValidationError):
        DataConfig(
            raw_path="x.csv",
            synthetic=SyntheticConfig(),
            target="Price",
            numeric_features=["Ram"],
            categorical_features=["Company"],
            test_size=1.5,
        )


def test_rejects_unknown_metric():
    """An unsupported AutoML metric is rejected by the Literal constraint."""
    with pytest.raises(ValidationError):
        ModelConfig(metric="banana")
