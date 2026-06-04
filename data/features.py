"""Feature engineering: turn the raw laptop CSV into clean model-ready columns.

The raw Kaggle dataset stores several fields as messy strings (RAM as "8GB", weight as
"1.37kg", screen resolution / CPU / memory as free text). This module parses them into
numeric and categorical features and drops physically invalid rows.
"""

from __future__ import annotations

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
    "Cpu_brand",
    "SSD",
    "HDD",
    "Gpu_brand",
    "Os",
    "Price",
]


def _cpu_brand(text: str) -> str:
    """Collapse the raw CPU string into a small set of brand/tier labels."""
    head = " ".join(str(text).split()[:3])
    if head in {"Intel Core i7", "Intel Core i5", "Intel Core i3"}:
        return head
    return "Other Intel" if str(text).split()[0] == "Intel" else "AMD"


def _parse_memory(text: str) -> pd.Series:
    """Split the raw memory string into total SSD and HDD capacity in GB."""
    cleaned = str(text).replace("GB", "").replace("TB", "000")
    ssd = hdd = 0
    for part in cleaned.split("+"):
        amount = int("".join(ch for ch in part if ch.isdigit()) or 0)
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
    df["Cpu_brand"] = df["Cpu"].apply(_cpu_brand)
    df[["SSD", "HDD"]] = df["Memory"].apply(_parse_memory)
    df["Gpu_brand"] = df["Gpu"].apply(lambda text: str(text).split()[0])
    df = df[df["Gpu_brand"] != "ARM"]
    df["Os"] = df["OpSys"].apply(_os_group)
    df = df.dropna(subset=["Ram", "Weight", "ppi"])
    return df[[column for column in ENGINEERED_COLUMNS if column in df.columns]]
