"""Staged retraining pipeline with data checks, model gates and artifact promotion."""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from config import AppConfig
from data.load import load_data
from model.schemas import DataValidationReport, ModelInfo, ModelValidationReport, RetrainingResult
from model.train import train


def validate_training_data(config: AppConfig) -> DataValidationReport:
    """Load and validate the current training dataset against the configured schema."""
    required = config.feature_columns + [config.data.target]
    source = "real" if Path(config.data.raw_path).exists() else "synthetic"
    try:
        df = load_data(config)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as ex:
        return DataValidationReport(
            ok=False,
            source=source,
            n_rows=0,
            min_rows=config.validation.min_rows,
            detail=str(ex),
        )

    missing = [column for column in required if column not in df.columns]
    extra = [column for column in df.columns if column not in required]
    null_counts = {column: int(df[column].isna().sum()) for column in required if column in df}
    enough_rows = len(df) >= config.validation.min_rows
    ok = not missing and enough_rows
    detail = "Training data passed validation."
    if missing:
        detail = f"Training data is missing required columns: {missing}"
    elif not enough_rows:
        detail = (
            f"Training data has {len(df)} rows; required minimum is {config.validation.min_rows}."
        )

    return DataValidationReport(
        ok=ok,
        source=source,
        n_rows=int(len(df)),
        min_rows=config.validation.min_rows,
        missing_columns=missing,
        extra_columns=extra,
        null_counts=null_counts,
        detail=detail,
    )


def validate_model_report(
    report: ModelInfo,
    config: AppConfig,
    min_r2: Optional[float] = None,
    max_mae: Optional[float] = None,
) -> ModelValidationReport:
    """Check model metrics against quality gates before artifact promotion."""
    r2_gate = config.validation.min_r2 if min_r2 is None else min_r2
    mae_gate = config.validation.max_mae if max_mae is None else max_mae
    checks = {
        "r2_is_finite": math.isfinite(report.r2),
        "mae_is_finite": math.isfinite(report.mae),
        "r2_above_minimum": report.r2 >= r2_gate,
        "mae_below_maximum": mae_gate is None or report.mae <= mae_gate,
        "row_count_above_minimum": report.n_rows >= config.validation.min_rows,
    }
    ok = all(checks.values())
    detail = "Model passed validation gates."
    if not ok:
        failed = [name for name, passed in checks.items() if not passed]
        detail = f"Model failed validation gates: {failed}"
    return ModelValidationReport(
        ok=ok, min_r2=r2_gate, max_mae=mae_gate, checks=checks, detail=detail
    )


def retrain_with_validation(
    config: AppConfig,
    time_budget_s: Optional[int] = None,
    min_r2: Optional[float] = None,
    max_mae: Optional[float] = None,
) -> RetrainingResult:
    """Train into a temporary directory and promote artifacts only after validation."""
    data_report = validate_training_data(config)
    if not data_report.ok:
        return RetrainingResult(
            status="failed",
            promoted=False,
            detail=data_report.detail,
            data_validation=data_report,
        )

    with tempfile.TemporaryDirectory(prefix="suml_retrain_") as tmp_dir:
        staged_config = config.model_copy(deep=True)
        staged_config.model.artifact_dir = tmp_dir
        if time_budget_s is not None:
            staged_config.model.time_budget_s = time_budget_s

        try:
            report = train(staged_config)
        except (RuntimeError, ValueError, OSError) as ex:
            return RetrainingResult(
                status="failed",
                promoted=False,
                detail=f"Training failed: {ex}",
                data_validation=data_report,
            )

        model_report = validate_model_report(report, staged_config, min_r2=min_r2, max_mae=max_mae)
        if not model_report.ok:
            return RetrainingResult(
                status="failed",
                promoted=False,
                detail=model_report.detail,
                data_validation=data_report,
                model_validation=model_report,
                model_info=report,
            )

        _promote_artifacts(staged_config, config)
        return RetrainingResult(
            status="succeeded",
            promoted=True,
            detail="Retraining finished and artifacts were promoted.",
            data_validation=data_report,
            model_validation=model_report,
            model_info=report,
        )


def _promote_artifacts(staged_config: AppConfig, target_config: AppConfig) -> None:
    """Copy staged artifacts into the configured artifact directory."""
    Path(target_config.model.artifact_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(staged_config.artifact_path, target_config.artifact_path)
    shutil.copy2(staged_config.metrics_path, target_config.metrics_path)
