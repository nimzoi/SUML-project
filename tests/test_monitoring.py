"""Tests for training-data profiling and drift reporting."""

import pandas as pd

from config import load_config
from data.monitoring import build_data_profile, compare_data_profiles


def _frame(ram=8, company="Dell"):
    """Return a tiny engineered dataframe compatible with the configured feature set."""
    return pd.DataFrame(
        [
            {
                "Ram": ram,
                "Weight": 1.6,
                "Inches": 15.6,
                "ppi": 141.2,
                "SSD": 256,
                "HDD": 0,
                "Touchscreen": 0,
                "Ips": 1,
                "Cpu_rank": 2,
                "Company": company,
                "TypeName": "Notebook",
                "Gpu_brand": "Intel",
                "Os": "Windows",
                "Price": 50000,
            },
            {
                "Ram": ram,
                "Weight": 1.8,
                "Inches": 15.6,
                "ppi": 141.2,
                "SSD": 512,
                "HDD": 0,
                "Touchscreen": 0,
                "Ips": 1,
                "Cpu_rank": 3,
                "Company": company,
                "TypeName": "Ultrabook",
                "Gpu_brand": "Intel",
                "Os": "Windows",
                "Price": 70000,
            },
        ]
    )


def test_build_data_profile_summarizes_configured_features():
    """The profile stores compact numeric and categorical distributions."""
    profile = build_data_profile(_frame(), load_config())
    assert profile.n_rows == 2
    assert profile.numeric["Ram"].mean == 8.0
    assert profile.categorical["Company"].frequencies["Dell"] == 1.0


def test_compare_identical_profiles_has_no_drift():
    """A profile compared with itself has zero material drift."""
    profile = build_data_profile(_frame(), load_config())
    report = compare_data_profiles(profile, profile)
    assert report.ok is True
    assert report.drifted_features == 0


def test_compare_profiles_flags_large_numeric_shift():
    """A large RAM shift is flagged as drift."""
    cfg = load_config()
    reference = build_data_profile(_frame(ram=8), cfg)
    current = build_data_profile(_frame(ram=64), cfg)
    report = compare_data_profiles(reference, current)
    ram_report = next(item for item in report.features if item.feature == "Ram")
    assert report.ok is False
    assert ram_report.drifted is True
