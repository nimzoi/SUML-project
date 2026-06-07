"""Tests for staged retraining and validation gates."""

from datetime import datetime, timezone
from pathlib import Path

import model.retraining as retraining
from config import load_config
from model.schemas import ModelInfo


def _model_info(cfg, r2=0.9, mae=1000.0):
    """Return a minimal ModelInfo suitable for validation tests."""
    return ModelInfo(
        mae=mae,
        rmse=mae * 1.5,
        r2=r2,
        best_estimator="fake",
        training_date=datetime.now(timezone.utc).isoformat(),
        n_rows=1300,
        data_source="real",
        feature_columns=cfg.feature_columns,
        target=cfg.data.target,
    )


def _tmp_config(tmp_path):
    """Return an isolated config with artifacts written under tmp_path."""
    cfg = load_config().model_copy(deep=True)
    cfg.model.artifact_dir = str(tmp_path / "artifacts")
    cfg.validation.min_rows = 1
    return cfg


def test_validate_training_data_current_config_ok():
    """The checked-in dataset satisfies the configured training-data gates."""
    report = retraining.validate_training_data(load_config())
    assert report.ok is True
    assert report.n_rows >= report.min_rows
    assert not report.missing_columns


def test_validate_model_report_rejects_low_quality_model(tmp_path):
    """A model below the R2 gate is rejected before promotion."""
    cfg = _tmp_config(tmp_path)
    report = retraining.validate_model_report(_model_info(cfg, r2=0.1), cfg, min_r2=0.7)
    assert report.ok is False
    assert report.checks["r2_above_minimum"] is False


def test_retrain_with_validation_promotes_staged_artifacts(tmp_path, monkeypatch):
    """A passing staged training run copies artifacts into the configured target dir."""
    cfg = _tmp_config(tmp_path)

    def _fake_train(staged_config):
        Path(staged_config.model.artifact_dir).mkdir(parents=True, exist_ok=True)
        staged_config.artifact_path.write_bytes(b"new-model")
        staged_config.metrics_path.write_text('{"ok": true}', encoding="utf-8")
        return _model_info(cfg, r2=0.9, mae=1000.0)

    monkeypatch.setattr(retraining, "train", _fake_train)
    result = retraining.retrain_with_validation(cfg, min_r2=0.7, max_mae=5000)
    assert result.status == "succeeded"
    assert result.promoted is True
    assert cfg.artifact_path.read_bytes() == b"new-model"
    assert cfg.metrics_path.exists()


def test_retrain_with_validation_does_not_promote_failed_model(tmp_path, monkeypatch):
    """A model that fails gates leaves the target artifact directory untouched."""
    cfg = _tmp_config(tmp_path)

    def _fake_train(staged_config):
        Path(staged_config.model.artifact_dir).mkdir(parents=True, exist_ok=True)
        staged_config.artifact_path.write_bytes(b"bad-model")
        staged_config.metrics_path.write_text('{"ok": false}', encoding="utf-8")
        return _model_info(cfg, r2=0.1, mae=1000.0)

    monkeypatch.setattr(retraining, "train", _fake_train)
    result = retraining.retrain_with_validation(cfg, min_r2=0.7, max_mae=5000)
    assert result.status == "failed"
    assert result.promoted is False
    assert not cfg.artifact_path.exists()
