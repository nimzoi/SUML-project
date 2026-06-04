# Data Card — Laptop Prices

## Source
- Kaggle laptop price dataset (campusx mirror), saved at `data/raw/laptop_data.csv`.
- 1303 raw rows × 12 columns; target = **Price** (currency: INR, as in the dataset).
- Committed to the repo for one-command reproducibility.

When the CSV is absent, `data/synthetic.py` deterministically generates data with the
**same engineered schema** (seeded), so the pipeline runs anywhere.

## Raw → engineered (the cleaning / feature-engineering layer)
The raw data stores several fields as messy strings. `data/features.py` parses them:

| Raw field | Engineered feature(s) |
|-----------|-----------------------|
| `Ram` = "8GB" | `Ram` (int) |
| `Weight` = "1.37kg" | `Weight` (float) |
| `ScreenResolution` = "IPS Panel … 1920x1080" | `Touchscreen` (0/1), `Ips` (0/1), `ppi` (float) |
| `Cpu` = "Intel Core i5 7200U 2.5GHz" | `Cpu_brand` (i3/i5/i7/Other Intel/AMD) |
| `Memory` = "256GB SSD + 1TB HDD" | `SSD` (GB), `HDD` (GB) |
| `Gpu` = "Nvidia GeForce MX150" | `Gpu_brand` (Intel/Nvidia/AMD) |
| `OpSys` | `Os` (Windows/Mac/Other) |

It also drops the index column and **drops rows with invalid/missing engineered values**
(physically impossible or unparseable) — real data cleaning.

## Engineered schema (target = `Price`)
- **Numeric:** `Ram`, `Weight`, `Inches`, `ppi`, `SSD`, `HDD`, `Touchscreen`, `Ips`
- **Categorical:** `Company`, `TypeName`, `Cpu_brand`, `Gpu_brand`, `Os`

## Target distribution
![Price distribution](img/target_hist.png)

## Top numeric feature vs price
![RAM vs price](img/feature_scatter.png)

## Baseline model
FLAML AutoML (stacked ensemble) with a **log-target** on a 20% holdout:
**MAE ≈ 9 600 · RMSE ≈ 14 600 · R² ≈ 0.85** on the original price scale
(≈ 0.88 measured on log-price). Top features (permutation importance): RAM, SSD, laptop
type, CPU tier. Live values are written to `model/artifacts/metrics.json` and served at
`GET /model-info`.
