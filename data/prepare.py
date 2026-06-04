"""Split the data and build the preprocessing ColumnTransformer."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import AppConfig


def split_data(
    df: pd.DataFrame, config: AppConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train/test feature frames and target series."""
    features = df[config.feature_columns]
    target = df[config.data.target]
    return train_test_split(
        features,
        target,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
    )


def build_preprocessor(config: AppConfig) -> ColumnTransformer:
    """Impute + one-hot categoricals; impute numerics (no scaling for tree models)."""
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric = SimpleImputer(strategy="median")
    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric, config.data.numeric_features),
            ("cat", categorical, config.data.categorical_features),
        ]
    )
    # Emit named DataFrames so fit and predict use consistent feature names
    # (avoids LightGBM's "X does not have valid feature names" warning).
    transformer.set_output(transform="pandas")
    return transformer
