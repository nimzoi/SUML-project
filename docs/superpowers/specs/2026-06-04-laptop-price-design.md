# Laptop Price Prediction — Design Spec

> Course: PJATK — Środowiska uruchomieniowe ML (SUML). Group project.
> Date: 2026-06-04. Presentation: 13.06.2026 or 27.06.2026.
> Note: this project started as "Food Delivery ETA" and pivoted to laptop-price
> prediction (for a higher, honest R² and a real data-cleaning showcase). The
> architecture/infra were reused unchanged; only the data-dependent layers changed.

## 1. Goal

A portable, reproducible ML application that predicts a laptop's **price** (INR) from its
specifications. Trained with **AutoML (FLAML)**, served as a **FastAPI** REST API with a
**Streamlit** UI, fully **containerized** with Docker Compose, driven by a single
`config.yaml`, with strict `data | model | app` separation. Runs with **no OS-level
configuration** — dependencies install automatically at container build.

## 2. Why this fits the SUML criteria (grading map)

| Criterion | Weight | How we satisfy it |
|-----------|-------:|-------------------|
| Code quality | 25% | PEP8, type hints, docstrings, black/isort, `pylint 10/10`, CI lint gate |
| Modularity | 20% | `data`/`model`/`app` packages; small single-purpose functions; config-driven |
| Portability | 20% | Docker Compose, pinned deps, slim non-root image, `packages.txt` (libgomp1), zero host config |
| Documentation | 20% | README + data card + FastAPI `/docs` + this spec + plan |
| ML model quality | 5% | FLAML AutoML (ensemble + log-target), R² ≈ 0.85 on the original price scale |
| Timeliness / organization | 10% | Small commits, CI, spec + plan |

The dataset is deliberately **messy** (specs stored as strings), which makes the
feature-engineering / cleaning layer a genuine part of the work — good for modularity and
code-quality marks.

## 3. Architecture

Three independent Python packages communicating through a saved artifact and HTTP.

```
SUML-project/
├── config.yaml                 # single source of truth (data + model + serving)
├── config.py                   # load + validate config into a typed Pydantic AppConfig
├── data/
│   ├── raw/laptop_data.csv     # raw Kaggle dataset (tracked)
│   ├── features.py             # raw strings -> engineered numeric/categorical features + cleaning
│   ├── synthetic.py            # deterministic synthetic generator (engineered schema)
│   ├── load.py                 # real CSV (-> engineer) or synthetic + structural validation
│   └── prepare.py              # split + ColumnTransformer (impute + one-hot)
├── model/
│   ├── train.py                # FLAML fit (ensemble + log-target) -> model.joblib + metrics.json
│   └── evaluate.py             # MAE / RMSE / R2
├── app/
│   ├── schemas.py              # Pydantic request/response models (enums)
│   ├── inference.py            # shared payload -> prediction helper (API + UI)
│   ├── api.py                  # FastAPI: /predict, /health, /model-info
│   └── ui.py                   # Streamlit front (calls API, or standalone local model)
├── tests/ · Dockerfile · docker-compose.yml · Makefile · packages.txt
├── pyproject.toml · .pylintrc · .github/workflows/ci.yml
└── docs/                       # data card, EDA, this spec, the plan
```

**Data flow:** `config.yaml` → `load` (real CSV → `features.engineer_features`, else
synthetic) → `prepare` (split + ColumnTransformer) → `train` (FLAML → Pipeline; write
`model.joblib` + `metrics.json`) → `api` loads the artifact → `ui` calls the api (or loads
the artifact directly in standalone mode).

## 4. Data layer

**Raw schema (12 cols):** `Unnamed: 0`, `Company`, `TypeName`, `Inches`,
`ScreenResolution`, `Cpu`, `Ram`, `Memory`, `Gpu`, `OpSys`, `Weight`, `Price` (target, INR).

**Feature engineering + cleaning (`data/features.py`):**

| Raw | Engineered |
|-----|------------|
| `Ram` = "8GB" | `Ram` (int) |
| `Weight` = "1.37kg" | `Weight` (float) |
| `ScreenResolution` = "IPS … 1920x1080" | `Touchscreen` (0/1), `Ips` (0/1), `ppi` (float) |
| `Cpu` = "Intel Core i5 7200U …" | `Cpu_brand` (i3/i5/i7/Other Intel/AMD) |
| `Memory` = "256GB SSD + 1TB HDD" | `SSD` (GB), `HDD` (GB) |
| `Gpu` = "Nvidia …" | `Gpu_brand` (Intel/Nvidia/AMD) |
| `OpSys` | `Os` (Windows/Mac/Other) |

Also drops the index column, drops `ARM` GPUs, and **drops rows with missing/invalid
engineered values** (`Ram`/`Weight`/`ppi`) — real cleaning.

**Engineered schema (target = `Price`):**
numeric `Ram, Weight, Inches, ppi, SSD, HDD, Touchscreen, Ips`;
categorical `Company, TypeName, Cpu_brand, Gpu_brand, Os`.

- **`load.py`**: real CSV → `engineer_features`; else synthetic (already engineered).
  Then `validate_schema` checks required columns exist + non-empty (per-cell nulls allowed
  — imputed in `prepare`).
- **`synthetic.py`**: deterministic (seeded) generator producing the engineered schema;
  price is a noisy linear function of specs; injects ~2% nulls so imputation is exercised.
- **`prepare.py`**: train/test split; `ColumnTransformer` — categoricals
  `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`,
  numerics `SimpleImputer(median)`; `set_output("pandas")` for consistent feature names.

## 5. Model layer

- **`train.py`**: `Pipeline([("prep", ColumnTransformer), ("model", <model>)])` where
  `<model>` is `AutoML()` or, when `log_target` is set, `TransformedTargetRegressor(AutoML(),
  func=log1p, inverse_func=expm1)` so the pipeline trains on log-price but **predicts the
  real price** (honest R² in currency units). FLAML settings (task, time budget, metric=r2,
  `estimator_list=[lgbm, rf, extra_tree]`, `ensemble=True`, seed) come from config.
- **Artifacts:** `model.joblib` (the Pipeline) + `metrics.json` (mae, rmse, r2,
  best_estimator, training_date, n_rows, data_source, feature_columns, target, and
  **permutation feature importance** — model-agnostic, works through the TTR + ensemble).
- **`evaluate.py`**: MAE/RMSE/R² on the original scale.
- **Result:** R² ≈ 0.85 (original INR scale; ≈ 0.88 on log-price), MAE ≈ 9.6k. Top
  features: RAM, SSD, type, CPU.

## 6. App layer (serving)

- **`schemas.py`**: Pydantic `PredictRequest` (snake_case fields, enums for Company /
  TypeName / CpuBrand / GpuBrand / Os, numeric fields with bounds), `PredictResponse(price)`,
  `HealthResponse`.
- **`inference.py`**: `REQUEST_TO_COLUMN` mapping + `to_feature_row` + `predict_price` —
  shared by the API and the standalone UI (DRY).
- **`api.py`**: FastAPI — `POST /predict` (Pydantic-validated → 422 on bad input, 503 if no
  model), `GET /health`, `GET /model-info`; model lazily loaded + cached; OpenAPI at `/docs`.
- **`ui.py`**: Streamlit form (brand/type/CPU/GPU/OS + RAM/SSD/HDD/weight/inches +
  touchscreen/IPS + resolution→PPI). Calls the API; **falls back to an in-process model**
  (trained once, cached) when no API is reachable → the same app runs standalone on
  Streamlit Community Cloud.

## 7. Configuration (`config.yaml`)

Single source of truth, validated on load. `data` (raw path, synthetic toggle/size/seed,
target, numeric/categorical feature lists, test split); `model` (task, time_budget_s,
metric, estimator_list, `ensemble`, `log_target`, artifact paths, seed); `api`/`ui`.
Swapping dataset / retuning = config edit, not code edit.

## 8. Engineering extras

- **Quality gates:** GitHub Actions CI (`pylint --fail-under=8` + `pytest`); black + isort;
  `.pylintrc`.
- **Portability:** `Dockerfile` (python:3.11-slim, libgomp1 for LightGBM, copy requirements
  first, non-root, trains the model at build), `.dockerignore`, `docker-compose.yml`
  (api + ui), `packages.txt` (libgomp1 for Streamlit Cloud), `Makefile`, `.gitattributes`
  (LF), pinned `requirements*.txt`.
- **Ops:** `metrics.json` + permutation importance, `/model-info`, logging, config validation.
- **Presentation aids:** data card + EDA plots (price histogram, RAM-vs-price scatter,
  feature importance), UI screenshot, `.pptx` deck.

## 9. Testing & verification

- `pytest` (28 tests): config, synthetic schema/determinism/nulls, load (real + synthetic +
  validation), prepare (split + impute/unknown handling), evaluate, train (artifact + sane
  prediction), schemas (valid/invalid), inference, api (`/health` `/predict` happy + 422 +
  `/model-info`), ui import.
- **Local verification (this machine):** native runs only — train, API, UI, tests,
  `pylint 10/10`, black/isort, `docker compose config`. **Docker build/run verified by the
  user on a second machine** (broken daemon here); a precise checklist is provided.

## 10. Deliverables

1. Runnable code, committed + pushed to `github.com/nimzoi/SUML-project` (short, unsigned
   commits).
2. `SUML_1.docx` proposal (Polish). **Needs the group's index numbers** (placeholder).
3. `slides.pptx` (Polish, ~10 min).
4. README + data card.

## 11. Out of scope (YAGNI)

No scheduler/drift detection, no DB, no auth, no cloud deploy automation, no xgboost/catboost
(measured to add nothing on this data and to bloat the image). Single dataset, single artifact.

## 12. Open items

- Group index numbers (for the proposal + title slide).
- Language for the slides (Polish).
