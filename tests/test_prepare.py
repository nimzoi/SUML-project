"""Tests for split and preprocessing."""

from config import load_config
from data.load import load_data
from data.prepare import build_preprocessor, split_data


def test_split_shapes_and_columns():
    """split_data partitions every row and keeps the configured feature columns."""
    cfg = load_config()
    df = load_data(cfg)
    x_train, x_test, y_train, y_test = split_data(df, cfg)
    assert len(x_train) + len(x_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)
    assert list(x_train.columns) == cfg.feature_columns


def test_preprocessor_handles_nulls_and_unknown_categories():
    """The preprocessor fits on train and transforms unseen test rows without error."""
    cfg = load_config()
    df = load_data(cfg)
    x_train, x_test, _, _ = split_data(df, cfg)
    pre = build_preprocessor(cfg)
    transformed_train = pre.fit_transform(x_train)
    assert transformed_train.shape[0] == len(x_train)
    transformed_test = pre.transform(x_test)
    assert transformed_test.shape[0] == len(x_test)
