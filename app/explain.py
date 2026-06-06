"""Per-prediction insights for the UI: price band, feature contributions, sensitivity.

Everything runs through the saved scikit-learn ``Pipeline`` (preprocessing + AutoML, with
the log-target inverse applied), so all numbers come back in real price units. The
contributions are a model-agnostic ablation: re-predict with one field reset to an
entry-level baseline and report the difference.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from pydantic import BaseModel

from app.inference import to_feature_row
from app.schemas import PredictRequest


class Contribution(BaseModel):
    """How much one feature adds to the price versus the baseline laptop."""

    label: str
    amount: float


class PriceBand(BaseModel):
    """A typical-error interval around a point estimate."""

    low: float
    high: float


# A deliberately entry-level reference laptop; contributions are measured against it.
BASELINE = PredictRequest(
    company="Acer",
    type_name="Notebook",
    inches=15.6,
    ram_gb=4,
    weight_kg=2.0,
    touchscreen=0,
    ips=0,
    ppi=100.0,
    cpu_brand="Other Intel",
    ssd_gb=0,
    hdd_gb=0,
    gpu_brand="Intel",
    os="Windows",
)

# Human-readable labels for each request field shown in the explanation.
FEATURE_LABELS: Dict[str, str] = {
    "company": "Marka",
    "type_name": "Typ",
    "cpu_brand": "Procesor",
    "gpu_brand": "Karta graficzna",
    "os": "System",
    "ram_gb": "RAM",
    "ssd_gb": "Dysk SSD",
    "hdd_gb": "Dysk HDD",
    "inches": "Przekątna ekranu",
    "ppi": "Gęstość ekranu (PPI)",
    "weight_kg": "Waga",
    "touchscreen": "Ekran dotykowy",
    "ips": "Panel IPS",
}


def _raw_price(model, request: PredictRequest) -> float:
    """Predict the (unrounded) price for one request through the full pipeline."""
    return float(model.predict(to_feature_row(request))[0])


def price_band(price: float, mae: float) -> PriceBand:
    """Return a typical-error band of ``price ± mae`` (lower bound clamped to 0)."""
    return PriceBand(low=max(0.0, price - mae), high=price + mae)


def explain_prediction(
    model, request: PredictRequest, baseline: PredictRequest = BASELINE
) -> List[Contribution]:
    """Estimate each feature's price contribution relative to an entry-level baseline.

    For every request field that differs from the baseline, re-predicts with that one
    field reset to its baseline value and records ``price - counterfactual``. A positive
    amount means the feature pushes the price up versus the baseline laptop. Returns the
    contributions sorted by descending absolute impact.
    """
    actual = _raw_price(model, request)
    contributions: List[Contribution] = []
    for field, label in FEATURE_LABELS.items():
        if getattr(request, field) == getattr(baseline, field):
            continue
        counterfactual = request.model_copy(update={field: getattr(baseline, field)})
        contributions.append(
            Contribution(label=label, amount=round(actual - _raw_price(model, counterfactual), 2))
        )
    contributions.sort(key=lambda contribution: abs(contribution.amount), reverse=True)
    return contributions


def price_sensitivity(model, request: PredictRequest, field: str, values: Sequence) -> Dict:
    """Map each candidate value of ``field`` to the predicted price (other fields fixed)."""
    prices = {}
    for value in values:
        variant = request.model_copy(update={field: value})
        prices[value] = round(_raw_price(model, variant), 2)
    return prices
