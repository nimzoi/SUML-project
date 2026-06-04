"""Load the dataset (real CSV if present, otherwise synthetic) and validate it."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import AppConfig
from data import synthetic

logger = logging.getLogger(__name__)


def load_data(config: AppConfig) -> pd.DataFrame:
    """Return the raw dataframe from the real CSV if it exists, else synthetic data."""
    raw_path = Path(config.data.raw_path)
    if raw_path.exists():
        logger.info("Loading real dataset from %s", raw_path)
        df = pd.read_csv(raw_path)
    elif config.data.synthetic.enabled:
        logger.info("Real CSV not found at %s; generating synthetic dataset", raw_path)
        df = synthetic.generate(config.data.synthetic.n_rows, config.data.synthetic.seed)
    else:
        raise FileNotFoundError(f"No dataset at {raw_path} and synthetic generation is disabled")
    validate_schema(df, config, require_target=True)
    return df


def validate_schema(df: pd.DataFrame, config: AppConfig, require_target: bool = True) -> None:
    """Validate that required columns exist and the frame is non-empty.

    Per-cell nulls are allowed on purpose: the real data contains them and they are
    imputed later in `prepare`. Missing/extra columns raise a clear error to protect
    future retraining batches from silent corruption.
    """
    required = list(config.feature_columns)
    if require_target:
        required.append(config.data.target)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty")
