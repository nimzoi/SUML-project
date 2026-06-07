"""Tests for Pandera dataframe contracts."""

import pandas as pd
import pytest
from pandera import errors as pa_errors

from config import load_config
from data.contracts import (
    validate_engineered_dataframe,
    validate_raw_dataframe,
)


def _raw_frame():
    """Return one valid raw laptop row."""
    return pd.DataFrame(
        [
            {
                "Company": "Dell",
                "TypeName": "Notebook",
                "Inches": 15.6,
                "ScreenResolution": "IPS Panel Full HD 1920x1080",
                "Cpu": "Intel Core i5 7200U 2.5GHz",
                "Ram": "8GB",
                "Memory": "256GB SSD",
                "Gpu": "Intel HD Graphics 620",
                "OpSys": "Windows 10",
                "Weight": "1.8kg",
                "Price": 50000,
            }
        ]
    )


def _engineered_frame():
    """Return one valid engineered laptop row."""
    return pd.DataFrame(
        [
            {
                "Ram": 8,
                "Weight": 1.8,
                "Inches": 15.6,
                "ppi": 141.21,
                "SSD": 256,
                "HDD": 0,
                "Touchscreen": 0,
                "Ips": 1,
                "Cpu_rank": 2,
                "Company": "Dell",
                "TypeName": "Notebook",
                "Gpu_brand": "Intel",
                "Os": "Windows",
                "Price": 50000,
            }
        ]
    )


def test_raw_contract_accepts_expected_csv_shape():
    """Raw contract accepts the checked CSV shape before feature engineering."""
    validated = validate_raw_dataframe(_raw_frame())
    assert validated.loc[0, "Company"] == "Dell"


def test_raw_contract_rejects_missing_required_column():
    """Raw contract rejects a CSV without a required source column."""
    with pytest.raises(pa_errors.SchemaErrors):
        validate_raw_dataframe(_raw_frame().drop(columns=["Memory"]))


def test_engineered_contract_accepts_model_schema():
    """Engineered contract accepts the model-ready schema used for training."""
    cfg = load_config()
    validated = validate_engineered_dataframe(_engineered_frame(), cfg)
    assert set(cfg.feature_columns).issubset(validated.columns)


def test_engineered_contract_rejects_invalid_binary_flag():
    """Engineered contract rejects binary indicator values outside 0/1."""
    cfg = load_config()
    frame = _engineered_frame()
    frame.loc[0, "Touchscreen"] = 2
    with pytest.raises(pa_errors.SchemaErrors):
        validate_engineered_dataframe(frame, cfg)
