"""Data profiling and drift checks for the training dataset."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from config import AppConfig
from model.schemas import (
    CategoricalFeatureProfile,
    DataDriftReport,
    DataProfile,
    DriftFeatureReport,
    NumericFeatureProfile,
)

MISSING_CATEGORY = "__MISSING__"
OTHER_CATEGORY = "__OTHER__"
NUMERIC_DRIFT_THRESHOLD = 0.35
CATEGORICAL_DRIFT_THRESHOLD = 0.25


def build_data_profile(
    df: pd.DataFrame, config: AppConfig, top_categories: int = 12
) -> DataProfile:
    """Summarize feature distributions in a compact, JSON-serializable profile."""
    return DataProfile(
        n_rows=int(len(df)),
        numeric=_numeric_profiles(df, config),
        categorical=_categorical_profiles(df, config, top_categories=top_categories),
    )


def compare_data_profiles(
    reference: DataProfile,
    current: DataProfile,
    numeric_threshold: float = NUMERIC_DRIFT_THRESHOLD,
    categorical_threshold: float = CATEGORICAL_DRIFT_THRESHOLD,
) -> DataDriftReport:
    """Compare two data profiles and flag features whose drift score exceeds a threshold."""
    features = []
    features.extend(_numeric_drift(reference, current, numeric_threshold))
    features.extend(_categorical_drift(reference, current, categorical_threshold))
    features.sort(key=lambda item: (not item.drifted, -item.score, item.feature))

    drifted = sum(item.drifted for item in features)
    detail = "Nie wykryto istotnego driftu danych."
    if drifted:
        detail = f"Liczba cech przekraczających progi driftu: {drifted}."

    return DataDriftReport(
        ok=drifted == 0,
        reference_rows=reference.n_rows,
        current_rows=current.n_rows,
        drifted_features=drifted,
        features=features,
        detail=detail,
    )


def _numeric_profiles(df: pd.DataFrame, config: AppConfig) -> Dict[str, NumericFeatureProfile]:
    """Return per-column numeric summary statistics."""
    profiles = {}
    row_count = max(len(df), 1)
    for column in config.data.numeric_features:
        if column not in df:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        observed = values.dropna()
        profiles[column] = NumericFeatureProfile(
            mean=_round(observed.mean()) if not observed.empty else 0.0,
            std=_round(observed.std(ddof=0)) if len(observed) > 1 else 0.0,
            min=_round(observed.min()) if not observed.empty else 0.0,
            max=_round(observed.max()) if not observed.empty else 0.0,
            missing_rate=_round(values.isna().sum() / row_count),
        )
    return profiles


def _categorical_profiles(
    df: pd.DataFrame, config: AppConfig, top_categories: int
) -> Dict[str, CategoricalFeatureProfile]:
    """Return compact categorical distributions with a retained top-k vocabulary."""
    profiles = {}
    row_count = max(len(df), 1)
    for column in config.data.categorical_features:
        if column not in df:
            continue
        values = df[column].where(df[column].notna(), MISSING_CATEGORY).astype(str)
        frequencies = values.value_counts(normalize=True).head(top_categories).to_dict()
        retained_mass = sum(frequencies.values())
        if retained_mass < 1.0:
            frequencies[OTHER_CATEGORY] = 1.0 - retained_mass
        profiles[column] = CategoricalFeatureProfile(
            frequencies={key: _round(value) for key, value in frequencies.items()},
            missing_rate=_round((values == MISSING_CATEGORY).sum() / row_count),
            unique_count=int(values.nunique()),
        )
    return profiles


def _numeric_drift(
    reference: DataProfile, current: DataProfile, threshold: float
) -> list[DriftFeatureReport]:
    """Calculate standardized mean-shift drift for numeric features."""
    reports = []
    for feature, reference_stats in reference.numeric.items():
        current_stats = current.numeric.get(feature)
        if current_stats is None:
            continue
        scale = max(abs(reference_stats.mean) * 0.1, reference_stats.std, 1.0)
        mean_shift = abs(current_stats.mean - reference_stats.mean) / scale
        missing_shift = abs(current_stats.missing_rate - reference_stats.missing_rate)
        score = _round(max(mean_shift, missing_shift))
        reports.append(
            DriftFeatureReport(
                feature=feature,
                kind="numeric",
                score=score,
                threshold=threshold,
                drifted=score > threshold,
                detail=(
                    f"średnia {reference_stats.mean} -> {current_stats.mean}; "
                    f"braki {reference_stats.missing_rate} -> {current_stats.missing_rate}"
                ),
            )
        )
    return reports


def _categorical_drift(
    reference: DataProfile, current: DataProfile, threshold: float
) -> list[DriftFeatureReport]:
    """Calculate total-variation drift for categorical feature distributions."""
    reports = []
    for feature, reference_stats in reference.categorical.items():
        current_stats = current.categorical.get(feature)
        if current_stats is None:
            continue
        score = max(
            _total_variation(reference_stats.frequencies, current_stats.frequencies),
            abs(current_stats.missing_rate - reference_stats.missing_rate),
        )
        rounded_score = _round(score)
        reports.append(
            DriftFeatureReport(
                feature=feature,
                kind="categorical",
                score=rounded_score,
                threshold=threshold,
                drifted=rounded_score > threshold,
                detail=(
                    f"liczba kategorii {reference_stats.unique_count} -> "
                    f"{current_stats.unique_count}; braki {reference_stats.missing_rate} -> "
                    f"{current_stats.missing_rate}"
                ),
            )
        )
    return reports


def _total_variation(reference: Dict[str, float], current: Dict[str, float]) -> float:
    """Return total variation distance between two discrete distributions."""
    categories = set(reference) | set(current)
    distances = (
        abs(reference.get(category, 0.0) - current.get(category, 0.0)) for category in categories
    )
    return 0.5 * sum(distances)


def _round(value: float) -> float:
    """Round profile values while keeping NaNs out of persisted JSON."""
    if pd.isna(value):
        return 0.0
    return round(float(value), 4)
