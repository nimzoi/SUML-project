"""Tests for the training pipeline."""

import joblib
import pandas as pd

from config import load_config
from model.train import train


def _fast_config(tmp_path):
    """Load the real config but with a tiny time budget and a temp artifact dir."""
    cfg = load_config()
    cfg.model.time_budget_s = 5
    cfg.model.artifact_dir = str(tmp_path)
    return cfg


def test_train_creates_artifact_and_metrics(tmp_path):
    """train writes model.joblib + metrics.json and reports sane metric values."""
    cfg = _fast_config(tmp_path)
    report = train(cfg)
    assert cfg.artifact_path.exists()
    assert cfg.metrics_path.exists()
    assert report.mae >= 0
    assert report.r2 <= 1.0
    assert report.data_source in {"real", "synthetic"}
    assert report.best_estimator


def test_saved_pipeline_predicts(tmp_path):
    """The persisted pipeline predicts a positive price from a raw feature row."""
    cfg = _fast_config(tmp_path)
    train(cfg)
    pipeline = joblib.load(cfg.artifact_path)
    row = pd.DataFrame(
        [
            {
                "Ram": 8,
                "Weight": 1.5,
                "Inches": 15.6,
                "ppi": 141.0,
                "SSD": 256,
                "HDD": 0,
                "Touchscreen": 0,
                "Ips": 1,
                "Company": "Dell",
                "TypeName": "Notebook",
                "Cpu_rank": 2,
                "Gpu_brand": "Intel",
                "Os": "Windows",
            }
        ]
    )
    prediction = pipeline.predict(row)
    assert prediction[0] > 0
