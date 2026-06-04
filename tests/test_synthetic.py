"""Tests for the synthetic data generator."""

from data.synthetic import COLUMNS, generate


def test_generate_shape_and_columns():
    df = generate(n_rows=50, seed=1)
    assert df.shape == (50, 9)
    assert list(df.columns) == COLUMNS


def test_generate_is_deterministic():
    assert generate(n_rows=20, seed=7).equals(generate(n_rows=20, seed=7))


def test_generate_injects_nulls():
    df = generate(n_rows=300, seed=3)
    assert df["Weather"].isnull().any()
    assert df["Courier_Experience_yrs"].isnull().any()
    assert df["Delivery_Time_min"].notnull().all()
    assert df["Order_ID"].notnull().all()


def test_target_is_positive_integer():
    df = generate(n_rows=100, seed=2)
    assert (df["Delivery_Time_min"] > 0).all()
    assert str(df["Delivery_Time_min"].dtype).startswith("int")
