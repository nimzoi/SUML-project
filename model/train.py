"""Train the model with FLAML AutoML and persist the artifact + metrics."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import joblib
from flaml import AutoML
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from config import AppConfig, load_config
from data.load import load_data
from data.prepare import build_preprocessor, split_data
from model.evaluate import regression_metrics

logger = logging.getLogger(__name__)


def train(config: AppConfig) -> Dict:
    """Run the full training pipeline and write model.joblib + metrics.json."""
    df = load_data(config)
    data_source = "real" if Path(config.data.raw_path).exists() else "synthetic"
    x_train, x_test, y_train, y_test = split_data(df, config)

    pipeline = Pipeline(steps=[("prep", build_preprocessor(config)), ("model", AutoML())])
    pipeline.fit(
        x_train,
        y_train,
        model__task=config.model.task,
        model__time_budget=config.model.time_budget_s,
        model__metric=config.model.metric,
        model__estimator_list=config.model.estimator_list,
        model__ensemble=config.model.ensemble,
        model__seed=config.model.seed,
        model__verbose=0,
    )

    metrics = regression_metrics(y_test, pipeline.predict(x_test))
    report = {
        **metrics,
        "best_estimator": pipeline.named_steps["model"].best_estimator,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df)),
        "data_source": data_source,
        "feature_columns": config.feature_columns,
        "target": config.data.target,
        "feature_importance": _feature_importance(pipeline, x_test, y_test, config.model.seed),
    }

    Path(config.model.artifact_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.artifact_path)
    with config.metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Saved model to %s | metrics=%s", config.artifact_path, metrics)
    return report


def _feature_importance(
    pipeline: Pipeline, x_test, y_test, seed: int, top_n: int = 15
) -> Dict[str, float]:
    """Model-agnostic permutation importance keyed by input feature.

    Works for any estimator (including stacked ensembles that lack
    feature_importances_): measures the drop in R2 when each input column is shuffled.
    """
    try:
        result = permutation_importance(
            pipeline, x_test, y_test, n_repeats=5, random_state=seed, scoring="r2"
        )
        ranked = sorted(
            zip(x_test.columns, result.importances_mean),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return {name: round(float(value), 4) for name, value in ranked[:top_n]}
    except (ValueError, AttributeError):
        return {}


def main() -> None:
    """CLI entry point: train with config.yaml and log the result."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(load_config())


if __name__ == "__main__":
    main()
