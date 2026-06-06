"""Per-prediction insights for the UI: price band, feature contributions, sensitivity.

Everything here runs through the saved scikit-learn ``Pipeline`` (preprocessing +
AutoML, with the log-target inverse applied), so all numbers come back in real price
units. The contributions are a simple, model-agnostic ablation: re-predict with one
field reset to an entry-level baseline and report the difference.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from app.inference import to_feature_row

# A deliberately entry-level reference laptop; contributions are measured against it.
BASELINE_PAYLOAD: Dict = {
    "company": "Acer",
    "type_name": "Notebook",
    "inches": 15.6,
    "ram_gb": 4,
    "weight_kg": 2.0,
    "touchscreen": 0,
    "ips": 0,
    "ppi": 100.0,
    "cpu_brand": "Other Intel",
    "ssd_gb": 0,
    "hdd_gb": 0,
    "gpu_brand": "Intel",
    "os": "Windows",
}

# Human-readable Polish labels for each request field shown in the explanation.
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


def _raw_price(model, payload: Dict) -> float:
    """Predict the (unrounded) price for one payload through the full pipeline."""
    return float(model.predict(to_feature_row(payload))[0])


def price_band(price: float, mae: float) -> Tuple[float, float]:
    """Return a ``(low, high)`` typical-error band of ``price ± mae`` (low clamped to 0)."""
    return (max(0.0, price - mae), price + mae)


def explain_prediction(model, payload: Dict, baseline: Dict = None) -> List[Tuple[str, float]]:
    """Estimate each feature's price contribution relative to an entry-level baseline.

    For every request field that differs from the baseline, re-predicts with that one
    field reset to its baseline value and records ``(label, price - counterfactual)``.
    A positive delta means the feature pushes the price up versus the baseline laptop.
    Returns the contributions sorted by descending absolute impact.
    """
    baseline = baseline or BASELINE_PAYLOAD
    actual = _raw_price(model, payload)
    contributions: List[Tuple[str, float]] = []
    for field, label in FEATURE_LABELS.items():
        if payload[field] == baseline[field]:
            continue
        counterfactual = dict(payload)
        counterfactual[field] = baseline[field]
        contributions.append((label, round(actual - _raw_price(model, counterfactual), 2)))
    contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return contributions


def price_sensitivity(model, payload: Dict, field: str, values: Sequence) -> Dict:
    """Map each candidate value of ``field`` to the predicted price (other fields fixed)."""
    prices = {}
    for value in values:
        variant = dict(payload)
        variant[field] = value
        prices[value] = round(_raw_price(model, variant), 2)
    return prices
