"""Tests for the raw-to-engineered feature parsing."""

import pandas as pd

from data.features import (
    ENGINEERED_COLUMNS,
    _cpu_brand,
    _os_group,
    _parse_memory,
    engineer_features,
)


def test_parse_memory_decimal_terabyte_not_inflated():
    """A decimal '1.0TB' drive parses to 1000 GB, not 10000 (naive TB->000 bug)."""
    ssd, hdd = _parse_memory("1.0TB Hybrid")
    assert ssd == 0
    assert hdd == 1000


def test_parse_memory_integer_terabyte():
    """A plain '2TB HDD' converts terabytes to 2000 GB."""
    ssd, hdd = _parse_memory("2TB HDD")
    assert ssd == 0
    assert hdd == 2000


def test_parse_memory_combined_ssd_and_hdd():
    """A '+'-joined string splits capacity across the SSD and HDD slots."""
    ssd, hdd = _parse_memory("256GB SSD +  1TB HDD")
    assert ssd == 256
    assert hdd == 1000


def test_parse_memory_flash_counts_as_ssd():
    """Flash storage is treated as SSD, leaving HDD at zero."""
    ssd, hdd = _parse_memory("128GB Flash Storage")
    assert ssd == 128
    assert hdd == 0


def test_cpu_brand_collapses_to_known_tiers():
    """The raw CPU string maps onto the small set of brand/tier labels."""
    assert _cpu_brand("Intel Core i7 7700HQ 2.8GHz") == "Intel Core i7"
    assert _cpu_brand("Intel Celeron Dual Core N3350") == "Other Intel"
    assert _cpu_brand("AMD A9-Series 9420 3GHz") == "AMD"


def test_os_group_buckets_operating_systems():
    """Raw OS strings collapse into Windows / Mac / Other."""
    assert _os_group("Windows 10") == "Windows"
    assert _os_group("macOS") == "Mac"
    assert _os_group("Linux") == "Other"


def test_engineer_features_produces_engineered_schema():
    """A messy raw frame is parsed into the full engineered schema with sane values."""
    raw = pd.DataFrame(
        [
            {
                "Unnamed: 0": 0,
                "Company": "Dell",
                "TypeName": "Notebook",
                "Inches": 15.6,
                "ScreenResolution": "IPS Panel Full HD 1920x1080",
                "Cpu": "Intel Core i5 7200U 2.5GHz",
                "Ram": "8GB",
                "Memory": "256GB SSD +  1.0TB Hybrid",
                "Gpu": "Intel HD Graphics 620",
                "OpSys": "Windows 10",
                "Weight": "1.8kg",
                "Price": 50000,
            }
        ]
    )

    eng = engineer_features(raw)

    assert list(eng.columns) == ENGINEERED_COLUMNS
    row = eng.iloc[0]
    assert row["Ram"] == 8
    assert row["SSD"] == 256
    assert row["HDD"] == 1000  # decimal-TB hybrid parsed correctly, not 10000
    assert row["Ips"] == 1
    assert row["Touchscreen"] == 0
    assert row["Cpu_rank"] == 2
    assert row["Gpu_brand"] == "Intel"
    assert row["Os"] == "Windows"
