"""Feature engineering: turn the raw laptop CSV into clean model-ready columns.

The raw Kaggle dataset stores several fields as messy strings (RAM as "8GB", weight as
"1.37kg", screen resolution / CPU / memory as free text). This module parses them into
numeric and categorical features and drops physically invalid rows.
"""

from __future__ import annotations

import re

import pandas as pd

ENGINEERED_COLUMNS = [
    "Company",
    "TypeName",
    "Inches",
    "Ram",
    "Weight",
    "Touchscreen",
    "Ips",
    "ppi",
    "Cpu_rank",
    "SSD",
    "HDD",
    "Gpu_brand",
    "Os",
    "Price",
]


CPU_RANK = {"Other Intel": 0, "AMD": 1, "Intel Core i3": 1, "Intel Core i5": 2, "Intel Core i7": 3}


def _cpu_brand(text: str) -> str:
    """Collapse the raw CPU string into a small set of brand/tier labels."""
    head = " ".join(str(text).split()[:3])
    if head in {"Intel Core i7", "Intel Core i5", "Intel Core i3"}:
        return head
    return "Other Intel" if str(text).split()[0] == "Intel" else "AMD"


def _parse_memory(text: str) -> pd.Series:
    """Split the raw memory string into total SSD and HDD capacity in GB.

    Each '+'-separated part looks like ``256GB SSD`` or ``1.0TB HDD``: the size is read
    as a number and terabytes are converted to gigabytes (x1000), so a decimal ``1.0TB``
    yields 1000 GB rather than 10000. Flash storage counts as SSD; hybrid drives as HDD.
    """
    ssd = hdd = 0
    for part in str(text).split("+"):
        match = re.search(r"([\d.]+)\s*(TB|GB)", part)
        if match is None:
            continue
        amount = round(float(match.group(1)) * (1000 if match.group(2) == "TB" else 1))
        if "SSD" in part or "Flash" in part:
            ssd += amount
        elif "HDD" in part or "Hybrid" in part:
            hdd += amount
    return pd.Series([ssd, hdd])


def _os_group(value: str) -> str:
    """Group the raw OS string into Windows / Mac / Other."""
    text = str(value)
    if "Windows" in text:
        return "Windows"
    if "Mac" in text or "macOS" in text:
        return "Mac"
    return "Other"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the raw laptop columns into the engineered schema; drop invalid rows."""
    df = df.copy()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["Ram"] = pd.to_numeric(df["Ram"].astype(str).str.replace("GB", ""), errors="coerce")
    df["Weight"] = pd.to_numeric(df["Weight"].astype(str).str.replace("kg", ""), errors="coerce")
    df["Touchscreen"] = df["ScreenResolution"].str.contains("Touchscreen").astype(int)
    df["Ips"] = df["ScreenResolution"].str.contains("IPS").astype(int)
    resolution = df["ScreenResolution"].str.extract(r"(\d+)x(\d+)").astype(float)
    df["ppi"] = ((resolution[0] ** 2 + resolution[1] ** 2) ** 0.5 / df["Inches"]).round(2)
    df["Cpu_rank"] = df["Cpu"].apply(lambda text: CPU_RANK[_cpu_brand(text)])
    df[["SSD", "HDD"]] = df["Memory"].apply(_parse_memory)
    df["Gpu_brand"] = df["Gpu"].apply(lambda text: str(text).split()[0])
    df = df[df["Gpu_brand"] != "ARM"]
    df["Os"] = df["OpSys"].apply(_os_group)
    df = df.dropna(subset=["Ram", "Weight", "ppi"])
    return df[[column for column in ENGINEERED_COLUMNS if column in df.columns]]
