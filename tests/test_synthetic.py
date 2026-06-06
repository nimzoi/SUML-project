"""Tests for the synthetic laptop data generator."""

from data.synthetic import COLUMNS, generate


def test_generate_shape_and_columns():
    """generate returns the requested row count and the full column schema."""
    df = generate(n_rows=50, seed=1)
    assert df.shape == (50, 14)
    assert list(df.columns) == COLUMNS


def test_generate_is_deterministic():
    """The same seed reproduces an identical frame."""
    assert generate(n_rows=20, seed=7).equals(generate(n_rows=20, seed=7))


def test_generate_injects_nulls():
    """Nulls land only in the imputation columns, never in the target or categoricals."""
    df = generate(n_rows=300, seed=3)
    assert df["Weight"].isnull().any()
    assert df["ppi"].isnull().any()
    assert df["Price"].notnull().all()
    assert df["Company"].notnull().all()


def test_price_is_positive_integer():
    """Generated prices are positive integers."""
    df = generate(n_rows=100, seed=2)
    assert (df["Price"] > 0).all()
    assert str(df["Price"].dtype).startswith("int")
