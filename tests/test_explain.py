"""Tests for per-prediction insights (price band, contributions, sensitivity)."""

import joblib

from app.explain import (
    BASELINE,
    Contribution,
    PriceBand,
    explain_prediction,
    price_band,
    price_sensitivity,
)
from app.schemas import PredictRequest


def _request():
    """Return a mid/high-spec laptop request (clearly above the entry-level baseline)."""
    return PredictRequest(
        company="Dell",
        type_name="Notebook",
        inches=15.6,
        ram_gb=16,
        weight_kg=1.6,
        touchscreen=0,
        ips=1,
        ppi=141.2,
        cpu_brand="Intel Core i7",
        ssd_gb=512,
        hdd_gb=0,
        gpu_brand="Nvidia",
        os="Windows",
    )


def test_price_band_brackets_estimate():
    """The band is the estimate plus/minus MAE."""
    band = price_band(50000.0, 9000.0)
    assert isinstance(band, PriceBand)
    assert band.low == 41000.0
    assert band.high == 59000.0


def test_price_band_clamps_low_to_zero():
    """The lower bound never goes negative."""
    band = price_band(5000.0, 9000.0)
    assert band.low == 0.0
    assert band.high == 14000.0


def test_explain_prediction_returns_sorted_contributions(trained_model):
    """Contributions are Contribution models sorted by descending absolute impact."""
    model = joblib.load(trained_model.artifact_path)
    contributions = explain_prediction(model, _request())
    assert contributions  # non-baseline laptop has something to explain
    assert all(isinstance(item, Contribution) for item in contributions)
    impacts = [abs(item.amount) for item in contributions]
    assert impacts == sorted(impacts, reverse=True)


def test_explain_prediction_skips_baseline_equal_fields(trained_model):
    """A laptop identical to the baseline has no contributions to explain."""
    model = joblib.load(trained_model.artifact_path)
    assert explain_prediction(model, BASELINE) == []


def test_price_sensitivity_ram_is_non_decreasing(trained_model):
    """More RAM never lowers the price (monotone constraint holds end-to-end)."""
    model = joblib.load(trained_model.artifact_path)
    prices = price_sensitivity(model, _request(), "ram_gb", [4, 8, 16, 32])
    ordered = [prices[ram] for ram in [4, 8, 16, 32]]
    assert ordered == sorted(ordered)
