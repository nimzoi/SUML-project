"""Tests for the training pipeline."""
import joblib
import pandas as pd

from config import load_config
from model.train import train


def _fast_config(tmp_path):
    cfg = load_config()
    cfg.model.time_budget_s = 5
    cfg.model.artifact_dir = str(tmp_path)
    return cfg


def test_train_creates_artifact_and_metrics(tmp_path):
    cfg = _fast_config(tmp_path)
    report = train(cfg)
    assert cfg.artifact_path.exists()
    assert cfg.metrics_path.exists()
    assert report["mae"] >= 0
    assert report["data_source"] in {"real", "synthetic"}
    assert "best_estimator" in report


def test_saved_pipeline_predicts(tmp_path):
    cfg = _fast_config(tmp_path)
    train(cfg)
    pipeline = joblib.load(cfg.artifact_path)
    row = pd.DataFrame(
        [
            {
                "Distance_km": 7.9,
                "Preparation_Time_min": 12,
                "Courier_Experience_yrs": 2.0,
                "Weather": "Clear",
                "Traffic_Level": "Medium",
                "Time_of_Day": "Afternoon",
                "Vehicle_Type": "Scooter",
            }
        ]
    )
    prediction = pipeline.predict(row)
    assert prediction[0] > 0
