# Laptop Price Prediction

![CI](https://github.com/nimzoi/SUML-project/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000)
![pylint](https://img.shields.io/badge/pylint-10.00%2F10-brightgreen)

Predicts a laptop's price (INR) from its specifications. Trained with **AutoML (FLAML)**,
served as a **FastAPI** REST API with a **Streamlit** UI, and fully **containerized** with
Docker Compose. Clean `data | model | app` separation, driven by a single `config.yaml`.

> Course: Środowiska uruchomieniowe ML (SUML), PJATK — group project.

![Laptop price UI](docs/img/ui.png)

## Business context

Pricing a laptop correctly matters for retailers, marketplaces and trade-in / valuation
tools: too high and it won't sell, too low and you lose margin. This project trains a
regression model that estimates price from brand, type, RAM, storage, screen, CPU/GPU and
OS, and exposes it through a `/predict` endpoint. Baseline: **R² ≈ 0.85** (original price
scale), MAE ≈ 9 600 INR.

## Architecture

Three independent packages, communicating through a saved artifact and HTTP:

- **data/** — load the raw CSV and **engineer + clean** it (`features.py`: parse "8GB"→8,
  "1.37kg"→1.37, screen resolution→PPI/touch/IPS, CPU/memory/GPU→brand & capacities), or
  generate synthetic data with the same engineered schema; then split + preprocess.
- **model/** — train via AutoML (FLAML) with a log-target and monotone constraints, and persist one
  scikit-learn `Pipeline` (`model.joblib`) plus `metrics.json`.
- **app/** — FastAPI service serving `/predict`, `/health`, `/model-info`; a Streamlit UI
  calls the API (with a standalone fallback that loads the model directly when no API is
  reachable). The UI shows a **price range**, a per-feature **"why this price"** breakdown
  (`app/explain.py`), and one-click **example presets**.

Everything is driven by `config.yaml`, so swapping the dataset or retuning AutoML is a
**config change, not a code change**.

## Repo structure

```
SUML-project/
├── config.yaml              # single source of truth: data + model + serving
├── requirements.txt         # pinned runtime deps
├── requirements-dev.txt     # + pytest, pylint, black, isort, httpx, matplotlib
├── Dockerfile               # slim, non-root, trains the model at build time
├── docker-compose.yml       # services: api (FastAPI) + ui (Streamlit)
├── Makefile · pyproject.toml · .pylintrc · packages.txt
├── .github/workflows/ci.yml # CI: pylint (>= 8) + pytest
├── config.py                # typed, validated config loader (Pydantic)
├── data/
│   ├── raw/laptop_data.csv  # dataset (tracked)
│   ├── features.py          # raw -> engineered features + cleaning
│   ├── synthetic.py         # deterministic synthetic generator (same schema)
│   ├── load.py              # load real CSV (+ engineer) or synthetic + validate
│   └── prepare.py           # split + ColumnTransformer (impute + one-hot)
├── model/
│   ├── train.py             # FLAML + log-target -> model.joblib + metrics.json
│   └── evaluate.py          # MAE / RMSE / R2
├── app/
│   ├── schemas.py           # Pydantic request/response models
│   ├── inference.py         # shared payload -> prediction helper (API + UI)
│   ├── api.py               # FastAPI: /predict, /health, /model-info
│   └── ui.py                # Streamlit front (API or standalone)
├── tests/                   # pytest: data, features, model, schemas, inference, explain, api, ui
└── docs/                    # data card, EDA plots, UI screenshot
```

## Requirements

- Python 3.11+
- Docker + Docker Compose (for the containerized path)

Dependencies are pinned in `requirements.txt` and installed at image build —
**no host-level configuration required**.

## Quickstart (Docker) — one command

```bash
git clone https://github.com/nimzoi/SUML-project.git
cd SUML-project
docker compose up --build
```

- API: http://localhost:8000 — interactive docs at `/docs`
- UI:  http://localhost:8501

The dataset is committed and the model is trained during the image build, so this runs
end-to-end on real data with no manual steps.

## Run natively (without Docker)

```bash
python -m pip install -r requirements-dev.txt
python -m model.train      # optional: a trained model is committed; run only to retrain
make api                   # or: uvicorn app.api:app --host 0.0.0.0 --port 8000
make ui                    # or: streamlit run app/ui.py
```

On Windows without `make`, use the explicit commands shown after each target.

## Configuration (`config.yaml`)

Single source of truth, validated on load. Key sections:

- `data` — dataset path, synthetic toggle + size + seed, target, numeric/categorical
  feature lists, test split.
- `model` — AutoML task, `time_budget_s`, metric, `estimator_list`, `ensemble`,
  `monotone_increasing` (guarantees more RAM/SSD/better CPU never lowers the price),
  `log_target` (train on log-price, predict real price), artifact paths, seed.
  **This is the AutoML config.**
- `api` / `ui` — host/port and the API URL the UI calls.

## Data

- Source: Kaggle laptop price dataset, committed at `data/raw/laptop_data.csv` (1303 raw
  rows). The raw strings are cleaned and feature-engineered in `data/features.py`.
- If the CSV is absent, a deterministic synthetic dataset with the same engineered schema
  is generated automatically. See [docs/data_card.md](docs/data_card.md).

## Retraining

Drop a new raw CSV (same columns) into `data/raw/` and run `python -m model.train`. The
loader engineers + validates it, and `model.joblib` + `metrics.json` are rebuilt — no code
changes. The app picks up the new artifact on restart.

## API

`POST /predict`

```json
{ "company": "Dell", "type_name": "Notebook", "inches": 15.6, "ram_gb": 8,
  "weight_kg": 1.6, "touchscreen": 0, "ips": 1, "ppi": 141.2,
  "cpu_brand": "Intel Core i5", "ssd_gb": 256, "hdd_gb": 0,
  "gpu_brand": "Intel", "os": "Windows" }
```

→ `{ "price": 55000.0 }`  *(example value, INR)*

- `GET /health` → `{ "status": "ok", "model_loaded": true }`
- `GET /model-info` → metrics + metadata (best estimator, training date, feature importance)

Invalid input is rejected with HTTP 422 (validated by Pydantic); requests before a model
is trained return HTTP 503.

## Testing & quality

```bash
make test     # pytest
make lint     # pylint >= 8  (currently 10.00/10)
make format   # black + isort
```

CI (GitHub Actions) runs `pylint --fail-under=8` and `pytest` on every push and PR.

## Deploy to Streamlit Cloud

The Streamlit UI is **standalone-capable**: if no API is reachable it loads the model
directly (training it once on first run, cached), so the same `app/ui.py` runs on
[Streamlit Community Cloud](https://share.streamlit.io) with no separate API:

1. Make sure the repo is public (or authorize Streamlit for a private repo).
2. *New app* → repo `nimzoi/SUML-project`, branch `main`, main file `app/ui.py` → *Deploy*.
3. Python 3.11+. Dependencies come from `requirements.txt`; `packages.txt` installs
   `libgomp1` (needed by LightGBM). The model trains once on first load (~60s), then is cached.

## Docs

- Instrukcja instalacji (PL): [docs/INSTRUKCJA.md](docs/INSTRUKCJA.md) — step-by-step setup, usage and retraining for end users.
- Data card: [docs/data_card.md](docs/data_card.md) — the dataset, cleaning/feature engineering, and model baseline.
