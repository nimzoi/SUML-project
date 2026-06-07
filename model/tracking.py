"""Optional MLflow tracking for model training runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from config import AppConfig
from model.schemas import ModelInfo

logger = logging.getLogger(__name__)


def log_training_run(config: AppConfig, report: ModelInfo) -> Dict[str, str]:
    """Log a training run to MLflow and return run metadata for metrics.json."""
    mlflow_config = config.tracking.mlflow
    if not mlflow_config.enabled:
        return {}

    try:
        import mlflow  # pylint: disable=import-outside-toplevel
    except ImportError:
        logger.warning("MLflow tracking is enabled, but mlflow is not installed.")
        return {}

    mlflow.set_tracking_uri(_tracking_uri(mlflow_config.tracking_uri))
    mlflow.set_experiment(mlflow_config.experiment_name)

    run_name = f"train-{report.training_date[:19].replace(':', '-')}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_training_params(config, report))
        mlflow.log_metrics({"mae": report.mae, "rmse": report.rmse, "r2": report.r2})
        mlflow.set_tags(
            {
                "data_source": report.data_source,
                "best_estimator": report.best_estimator,
                "target": report.target,
            }
        )
        _log_existing_artifact(mlflow, config.artifact_path, "model")
        _log_existing_artifact(mlflow, Path("config.yaml"), "config")

    tracking_uri = mlflow.get_tracking_uri()
    logger.info("Logged MLflow run %s to %s", run.info.run_id, tracking_uri)
    return {"mlflow_run_id": run.info.run_id, "mlflow_tracking_uri": tracking_uri}


def _training_params(config: AppConfig, report: ModelInfo) -> Dict[str, object]:
    """Return compact run parameters useful in MLflow UI."""
    return {
        "task": config.model.task,
        "metric": config.model.metric,
        "time_budget_s": config.model.time_budget_s,
        "estimator_list": ",".join(config.model.estimator_list),
        "ensemble": config.model.ensemble,
        "log_target": config.model.log_target,
        "monotone_increasing": ",".join(config.model.monotone_increasing),
        "test_size": config.data.test_size,
        "random_state": config.data.random_state,
        "n_rows": report.n_rows,
        "n_features": len(report.feature_columns),
    }


def _tracking_uri(uri: str) -> str:
    """Normalize a local tracking path to a file URI; keep remote URIs unchanged."""
    if "://" in uri or uri.startswith("file:"):
        return uri
    return Path(uri).resolve().as_uri()


def _log_existing_artifact(mlflow, path: Path, artifact_path: str) -> None:
    """Log one artifact if it exists, without making tracking mandatory."""
    if path.exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
