# Food Delivery ETA — Design Spec

> Course: PJATK — Środowiska uruchomieniowe ML (SUML). Group project.
> Date: 2026-06-04. Presentation: 13.06.2026 or 27.06.2026.

## 1. Goal

A portable, reproducible ML application that predicts food-delivery time (minutes)
from order/route/context features. Trained with **AutoML (FLAML)**, served as a
**FastAPI** REST service with a **Streamlit** UI, fully **containerized** with
Docker Compose, driven by a single `config.yaml`, with strict `data | model | app`
separation. Must run with **no OS-level configuration** — dependencies install
automatically at container build.

## 2. Why this fits the SUML criteria (grading map)

| Criterion | Weight | How we satisfy it |
|-----------|-------:|-------------------|
| Code quality | 25% | PEP8, type hints, docstrings, black/isort, `pylint >= 8`, CI lint gate |
| Modularity | 20% | `data`/`model`/`app` packages; small single-purpose functions; config-driven |
| Portability | 20% | Docker Compose, pinned `requirements.txt`, slim non-root image, zero host config |
| Documentation | 20% | README + code comments + FastAPI `/docs` (OpenAPI) + data card |
| ML model quality | 5% | FLAML AutoML, reported MAE/RMSE/R² |
| Timeliness / organization | 10% | Git history with small commits, CI, this spec + plan |

The ML is intentionally simple: the course grades *runtime environments*, so effort
goes into reproducibility and clean separation, not chasing model accuracy.

## 3. Architecture

Three independent Python packages communicating through files (artifacts) and HTTP.

```
food-delivery-eta/
├── config.yaml                 # single source of truth (data + model + serving)
├── requirements.txt            # pinned versions
├── Dockerfile                  # slim, non-root, healthcheck
├── docker-compose.yml          # services: api (FastAPI) + ui (Streamlit)
├── .dockerignore
├── .pylintrc                   # tuned to keep pylint >= 8
├── Makefile                    # train / api / ui / test / lint / format / docker
├── pyproject.toml              # black + isort config
├── .github/workflows/ci.yml    # pylint + pytest on push/PR
├── data/
│   ├── __init__.py
│   ├── raw/.gitkeep            # real Kaggle CSV lands here (git-ignored)
│   ├── synthetic.py           # deterministic synthetic generator (same schema)
│   ├── load.py                # read real CSV if present, else synthetic; validate schema
│   └── prepare.py             # split + ColumnTransformer (encode)
├── model/
│   ├── __init__.py
│   ├── train.py               # FLAML fit -> Pipeline; save model.joblib + metrics.json
│   ├── evaluate.py            # MAE / RMSE / R2 helpers
│   └── artifacts/.gitkeep     # model.joblib + metrics.json (git-ignored)
├── app/
│   ├── __init__.py
│   ├── schemas.py             # Pydantic request/response models
│   ├── api.py                 # FastAPI: /predict /health /model-info
│   └── ui.py                  # Streamlit front, calls the API
├── docs/
│   └── data_card.md           # dataset description + EDA plots (presentation aid)
├── tests/
│   ├── test_data.py           # synthetic schema, validation, split shapes
│   ├── test_model.py          # train tiny budget -> artifact loads, predict in range
│   └── test_api.py            # TestClient: /health, /predict happy + 422 paths
└── README.md
```

**Data flow:** `config.yaml` → `load` (real CSV or synthetic) → `prepare`
(split + fit ColumnTransformer) → `train` (FLAML → Pipeline; write `model.joblib`
+ `metrics.json`) → `api` loads the artifact → `ui` calls the api.

## 4. Data layer

**Schema (target = `Delivery_Time_min`)** — confirmed against the real Kaggle file
`denkuznetz/food-delivery-time-prediction` (1000 rows × 9 cols, present at
`data/raw/Food_Delivery_Times.csv`). **Real data has missing values:** 30 nulls each in
`Weather`, `Traffic_Level`, `Time_of_Day`, `Courier_Experience_yrs`.

| Column | Type | Notes |
|--------|------|-------|
| Order_ID | int | dropped before training |
| Distance_km | float | numeric |
| Weather | cat | Clear/Rainy/Snowy/Foggy/Windy |
| Traffic_Level | cat | Low/Medium/High |
| Time_of_Day | cat | Morning/Afternoon/Evening/Night |
| Vehicle_Type | cat | Bike/Scooter/Car |
| Preparation_Time_min | int | numeric |
| Courier_Experience_yrs | float | numeric |
| Delivery_Time_min | int | **target** |

- **`load.py`**: if `data/raw/<file>` exists → read it; else → generate synthetic via
  `synthetic.py`. Either way, **validate structure** (required columns present, dtypes
  coercible, target present for training) but **allow per-cell nulls** (the real data has
  them; imputation happens in `prepare`). Missing/extra columns raise a clear error —
  this protects future retraining batches from silent corruption.
- **`synthetic.py`**: deterministic (`seed` from config). Realistic distributions and a
  plausible signal: base time grows with distance, prep time, traffic, bad weather;
  drops slightly with courier experience; plus Gaussian noise. Injects ~3% nulls in the
  same columns as the real data, so the imputation path is exercised even without the real
  CSV. This is the schema source of truth when no real CSV is present.
- **`prepare.py`**: train/test split (config `test_size`, `random_state`); build a
  `ColumnTransformer` — categoricals: `SimpleImputer(strategy="most_frequent")` →
  `OneHotEncoder(handle_unknown="ignore")`; numerics: `SimpleImputer(strategy="median")`
  (no scaling — tree models do not need it). Returns split data + unfitted
  transformer (fitting happens in `train` so the fitted transformer ships in the artifact).

**Hybrid + retraining story:** real CSV and synthetic data go through the *same*
prepare pipeline. "Next batch of data" = drop a new CSV in `data/raw/` and re-run
`python -m model.train`. We train once now, but the architecture is retraining-ready;
no scheduler/MLOps (explicitly out of scope — YAGNI for this course).

## 5. Model layer

- **`train.py`**: `Pipeline([("prep", ColumnTransformer), ("model", AutoML)])`.
  Fit the transformer on train, fit FLAML on transformed data with settings from config
  (`task=regression`, `time_budget_s`, `metric`, `estimator_list=[lgbm, rf, extra_tree]`,
  `seed`). Assemble the fitted steps into one `Pipeline` and persist as `model.joblib`.
- **Artifacts:** `model.joblib` (the Pipeline) + `metrics.json` containing:
  `mae`, `rmse`, `r2`, `best_estimator`, `feature_importance` (if available),
  `training_date`, `n_rows`, `data_source` (real|synthetic), `schema`. This doubles as
  model metadata for `/model-info`.
- **`evaluate.py`**: pure functions computing MAE/RMSE/R² from y_true/y_pred.

## 6. App layer (serving)

Two services (chosen over single-Streamlit and custom-frontend alternatives):

- **`api.py` (FastAPI):**
  - `POST /predict` — body validated by Pydantic; builds a one-row DataFrame, runs
    `pipeline.predict`, returns `{ "eta_minutes": float }`. Returns 503 if no model
    artifact is loaded, 422 on invalid input (free via Pydantic).
  - `GET /health` — liveness + whether a model is loaded.
  - `GET /model-info` — serves `metrics.json` (metrics, best estimator, training date,
    data source). Powers the retraining narrative and the demo.
  - Model loaded once at startup; interactive docs at `/docs` (OpenAPI, free).
- **`schemas.py` (Pydantic):** `PredictRequest` (typed fields with example values +
  enums for categoricals so `/docs` is self-documenting) and `PredictResponse`.
- **`ui.py` (Streamlit):** input widgets matching the schema → calls the API → shows
  predicted ETA; a second panel shows `/model-info` (metrics + feature importance bar
  chart). API URL comes from config/env (`http://api:8000` in compose, `localhost:8000`
  native).

## 7. Configuration (`config.yaml`)

Single source of truth. Sections: `data` (raw path, synthetic on/off + n_rows + seed,
target, numeric_features, categorical_features, test_size, random_state),
`model` (task, time_budget_s, metric, estimator_list, artifact paths, seed),
`api` (host, port), `ui` (api_url). Validated on load (clear error if malformed) so a
bad config fails fast rather than mid-training. Swapping dataset/retuning = config edit,
not code edit.

## 8. Engineering extras (all four bundles selected)

- **Quality gates:** GitHub Actions CI runs `pylint` (fail under 8) + `pytest` on push/PR;
  `black` + `isort` via `pyproject.toml`; `.pylintrc` tuned.
- **Portability:** `Dockerfile` (python:3.11-slim, copy requirements first for layer
  caching, non-root user, `HEALTHCHECK`), `.dockerignore`, `docker-compose.yml`
  (api + ui), `Makefile` with `train/api/ui/test/lint/format/docker` + documented
  `python -m ...` fallbacks (cross-platform). The canonical dataset
  `data/raw/Food_Delivery_Times.csv` is tracked in git so clone → run works on real data;
  `.gitignore` excludes any other `data/raw/*` and `model/artifacts/*`.
- **Ops/retraining:** `metrics.json` + model metadata, `/model-info` endpoint, Python
  `logging` (no prints), config validation.
- **Presentation aids:** `data_card.md` (dataset description + a few EDA plots) and
  feature-importance display in the UI.

## 9. Testing & verification

- `pytest`: data validation/synthetic/split, train-with-tiny-budget → artifact loads and
  predicts a sane value, API `/health` + `/predict` (happy path + 422) via `TestClient`.
- **Local verification (this machine):** native runs only — train, API, UI import,
  tests, `pylint >= 8`, and `docker compose config` (syntax). **Docker build/run is NOT
  verifiable here** (Docker daemon broken on the dev machine); the user tests
  `git pull` → `docker compose up --build` on a second machine. A short checklist will
  be provided; no claim of a tested container path will be made locally.

## 10. Deliverables

1. Runnable code (all of the above), committed and pushed to
   `github.com/nimzoi/SUML-project` with short, unsigned commit messages.
2. `SUML_1.docx` project proposal filled in (Polish). **Needs: group index numbers.**
3. `.pptx` presentation slides (~10 min) — problem, architecture, AutoML, live-demo
   script, results.
4. Updated `README.md` (run instructions, input/output format, retraining note).

## 11. Out of scope (YAGNI)

No scheduler/cron retraining, no drift detection, no database, no auth, no cloud deploy,
no xgboost/catboost. Single dataset, single model artifact.

## 12. Open items

- Group index numbers (for the proposal).
- Language for the slides (assume Polish unless told otherwise).
