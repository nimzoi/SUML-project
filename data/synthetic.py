"""Generate a synthetic laptop dataset matching the engineered schema."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from data.features import CPU_RANK

COLUMNS: List[str] = [
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

COMPANIES = ["Dell", "HP", "Lenovo", "Asus", "Acer", "Apple", "MSI", "Toshiba", "Samsung", "Razer"]
TYPES = ["Notebook", "Ultrabook", "Gaming", "2 in 1 Convertible", "Workstation", "Netbook"]
CPU_BRANDS = ["Intel Core i3", "Intel Core i5", "Intel Core i7", "Other Intel", "AMD"]
GPU_BRANDS = ["Intel", "Nvidia", "AMD"]
OSES = ["Windows", "Mac", "Other"]

_CPU_PRICE = {
    "Intel Core i3": 5000,
    "Intel Core i5": 15000,
    "Intel Core i7": 32000,
    "Other Intel": 3000,
    "AMD": 9000,
}
_GPU_PRICE = {"Intel": 0, "Nvidia": 28000, "AMD": 12000}
_TYPE_PRICE = {
    "Netbook": -8000,
    "Notebook": 0,
    "2 in 1 Convertible": 12000,
    "Ultrabook": 22000,
    "Gaming": 30000,
    "Workstation": 42000,
}
_BRAND_PRICE = {
    "Apple": 45000,
    "Razer": 35000,
    "MSI": 18000,
    "Dell": 4000,
    "HP": 2000,
    "Lenovo": 3000,
    "Asus": 3000,
    "Acer": -2000,
    "Toshiba": 0,
    "Samsung": 6000,
}
_NULL_COLUMNS = ["Weight", "ppi"]


def generate(n_rows: int = 1300, seed: int = 42) -> pd.DataFrame:  # pylint: disable=too-many-locals
    """Return a DataFrame with the same engineered columns/types as the real dataset.

    Price is a noisy linear function of specs (RAM, storage, screen density, CPU/GPU tier,
    chassis type and brand premium). The CPU is stored as an ordinal rank (i3<i5<i7).
    About 2% of cells in a couple of columns are set to NaN so the imputation path is
    exercised even without the real CSV.
    """
    rng = np.random.default_rng(seed)
    company = rng.choice(COMPANIES, n_rows)
    type_name = rng.choice(TYPES, n_rows)
    inches = np.round(rng.uniform(11.0, 17.3, n_rows), 1)
    ram = rng.choice([4, 8, 12, 16, 32], n_rows, p=[0.15, 0.4, 0.1, 0.25, 0.1])
    weight = np.round(rng.uniform(1.0, 3.2, n_rows), 2)
    touchscreen = rng.integers(0, 2, n_rows)
    ips = rng.integers(0, 2, n_rows)
    ppi = np.round(rng.uniform(90.0, 280.0, n_rows), 1)
    cpu = rng.choice(CPU_BRANDS, n_rows)
    ssd = rng.choice([0, 128, 256, 512, 1000], n_rows, p=[0.2, 0.25, 0.3, 0.2, 0.05])
    hdd = rng.choice([0, 500, 1000, 2000], n_rows, p=[0.6, 0.2, 0.15, 0.05])
    gpu = rng.choice(GPU_BRANDS, n_rows, p=[0.5, 0.35, 0.15])
    operating_system = rng.choice(OSES, n_rows, p=[0.8, 0.1, 0.1])

    price = (
        18000.0
        + ram * 2200
        + ssd * 38
        + hdd * 4
        + ppi * 90
        + np.array([_CPU_PRICE[c] for c in cpu])
        + np.array([_GPU_PRICE[g] for g in gpu])
        + np.array([_TYPE_PRICE[t] for t in type_name])
        + np.array([_BRAND_PRICE[b] for b in company])
        + touchscreen * 7000
        + ips * 4000
        + rng.normal(0.0, 12000.0, n_rows)
    )
    price = np.clip(np.round(price), 9000, None).astype(int)

    df = pd.DataFrame(
        {
            "Company": company,
            "TypeName": type_name,
            "Inches": inches,
            "Ram": ram,
            "Weight": weight,
            "Touchscreen": touchscreen,
            "Ips": ips,
            "ppi": ppi,
            "Cpu_rank": [CPU_RANK[c] for c in cpu],
            "SSD": ssd,
            "HDD": hdd,
            "Gpu_brand": gpu,
            "Os": operating_system,
            "Price": price,
        }
    )
    _inject_nulls(df, rng, frac=0.02)
    return df


def _inject_nulls(df: pd.DataFrame, rng: np.random.Generator, frac: float) -> None:
    """Set a fraction of cells to NaN in a couple of numeric columns."""
    n_null = int(len(df) * frac)
    for column in _NULL_COLUMNS:
        idx = rng.choice(len(df), size=n_null, replace=False)
        df.loc[idx, column] = np.nan
