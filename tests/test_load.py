"""Tests for dataset loading and validation."""
import pandas as pd
import pytest

from config import load_config
from data.load import load_data, validate_schema


def test_load_real_csv_when_present():
    cfg = load_config()  # data/raw/Food_Delivery_Times.csv exists in the repo
    df = load_data(cfg)
    assert cfg.data.target in df.columns
    assert len(df) > 0


def test_synthetic_fallback_when_csv_missing():
    cfg = load_config()
    cfg.data.raw_path = "data/raw/__missing__.csv"
    df = load_data(cfg)
    assert len(df) == cfg.data.synthetic.n_rows


def test_validate_schema_raises_on_missing_column():
    cfg = load_config()
    with pytest.raises(ValueError):
        validate_schema(pd.DataFrame({"foo": [1, 2]}), cfg)


def test_validate_schema_allows_nulls():
    cfg = load_config()
    df = load_data(cfg)
    df.loc[df.index[:5], cfg.data.categorical_features[0]] = None
    validate_schema(df, cfg)
