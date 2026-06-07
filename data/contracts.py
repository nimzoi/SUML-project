"""Pandera dataframe contracts for raw and engineered training data."""

from __future__ import annotations

from typing import List

import pandas as pd
import pandera.pandas as pa

from config import AppConfig

RAW_TEXT_COLUMNS = [
    "Company",
    "TypeName",
    "ScreenResolution",
    "Cpu",
    "Ram",
    "Memory",
    "Gpu",
    "OpSys",
    "Weight",
]


def raw_laptop_schema(target: str = "Price") -> pa.DataFrameSchema:
    """Return the schema expected from the raw laptop CSV before feature engineering."""
    columns = {column: pa.Column(str, nullable=False, coerce=True) for column in RAW_TEXT_COLUMNS}
    columns["Inches"] = pa.Column(float, pa.Check.gt(0), nullable=False, coerce=True)
    columns[target] = pa.Column(float, pa.Check.gt(0), nullable=False, coerce=True)
    return pa.DataFrameSchema(columns, strict=False, coerce=True)


def engineered_laptop_schema(config: AppConfig, require_target: bool = True) -> pa.DataFrameSchema:
    """Return the schema expected after feature engineering and before preprocessing."""
    columns = {}
    for column in config.data.numeric_features:
        checks = _numeric_checks(column)
        columns[column] = pa.Column(float, checks=checks, nullable=True, coerce=True)
    for column in config.data.categorical_features:
        columns[column] = pa.Column(str, nullable=True)
    if require_target:
        columns[config.data.target] = pa.Column(float, pa.Check.gt(0), nullable=False, coerce=True)
    return pa.DataFrameSchema(columns, strict=False, coerce=True)


def validate_raw_dataframe(df: pd.DataFrame, target: str = "Price") -> pd.DataFrame:
    """Validate raw CSV columns and value ranges with Pandera."""
    return raw_laptop_schema(target).validate(df, lazy=True)


def validate_engineered_dataframe(
    df: pd.DataFrame, config: AppConfig, require_target: bool = True
) -> pd.DataFrame:
    """Validate engineered model input columns and value ranges with Pandera."""
    return engineered_laptop_schema(config, require_target=require_target).validate(df, lazy=True)


def schema_error_messages(error: Exception, limit: int = 5) -> List[str]:
    """Format the most useful Pandera failure cases for API and CLI reports."""
    failure_cases = getattr(error, "failure_cases", None)
    if isinstance(failure_cases, pd.DataFrame) and not failure_cases.empty:
        return [str(item) for item in failure_cases.head(limit).to_dict("records")]
    return [str(error)]


def _numeric_checks(column: str):
    """Return value checks for known engineered numeric columns."""
    if column in {"Touchscreen", "Ips"}:
        return pa.Check.isin([0, 1])
    if column in {"Weight", "Inches", "ppi"}:
        return pa.Check.gt(0)
    return pa.Check.ge(0)
