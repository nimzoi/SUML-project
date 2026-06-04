"""Train the model with FLAML AutoML and persist the artifact + metrics."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from flaml import AutoML
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from config import AppConfig, load_config
from data.load import load_data
from data.prepare import build_preprocessor, split_data
from model.evaluate import regression_metrics

logger = logging.getLogger(__name__)


def _monotone_constraints(columns, increasing) -> str:
    """Build a LightGBM monotone_constraints string (+1 for the increasing numeric features)."""
    wanted = set(increasing)
    return ",".join(
        "1" if (col.startswith("num__") and col.split("__", 1)[1] in wanted) else "0"
        for col in columns
    )


def train(config: AppConfig) -> Dict:  # pylint: disable=too-many-locals
    """Run the full training pipeline and write model.joblib + metrics.json."""
    df = load_data(config)
    data_source = "real" if Path(config.data.raw_path).exists() else "synthetic"
    x_train, x_test, y_train, y_test = split_data(df, config)

    preprocessor = build_preprocessor(config)
    x_train_t = preprocessor.fit_transform(x_train)

    fit_kwargs = {
        "task": config.model.task,
        "time_budget": config.model.time_budget_s,
        "metric": config.model.metric,
        "estimator_list": config.model.estimator_list,
        "ensemble": config.model.ensemble,
        "seed": config.model.seed,
        "verbose": 0,
    }
    if config.model.monotone_increasing:
        constraints = _monotone_constraints(x_train_t.columns, config.model.monotone_increasing)
        fit_kwargs["custom_hp"] = {"lgbm": {"monotone_constraints": {"domain": constraints}}}

    model_step = AutoML()
    if config.model.log_target:
        # Train on log(price) but predict on the original scale (honest R2 in real units).
        model_step = TransformedTargetRegressor(
            regressor=AutoML(), func=np.log1p, inverse_func=np.expm1
        )
    model_step.fit(x_train_t, y_train, **fit_kwargs)

    pipeline = Pipeline(steps=[("prep", preprocessor), ("model", model_step)])
    metrics = regression_metrics(y_test, pipeline.predict(x_test))
    automl = getattr(model_step, "regressor_", model_step)
    report = {
        **metrics,
        "best_estimator": automl.best_estimator,
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

    Works for any estimator: measures the drop in R2 when each input column is shuffled.
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
