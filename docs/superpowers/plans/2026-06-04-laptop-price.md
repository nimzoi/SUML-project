# Laptop Price Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portable, reproducible app that predicts a laptop's price (INR) from its specifications, via FLAML AutoML, served by FastAPI + Streamlit, containerized with Docker Compose.

**Architecture:** Strict `data | model | app` packages driven by `config.yaml`. `data` loads the raw CSV and **engineers + cleans** it (or generates synthetic data with the same engineered schema), then splits/preprocesses; `model` trains via FLAML (ensemble + log-target) and persists one `sklearn.Pipeline` artifact + `metrics.json`; `app` serves `/predict`, `/health`, `/model-info` over FastAPI with a Streamlit UI that also runs standalone.

**Tech Stack:** Python 3.11+, pandas, scikit-learn, FLAML + LightGBM, FastAPI + uvicorn, Streamlit, Pydantic, PyYAML, joblib; pytest, pylint, black, isort; Docker Compose; GitHub Actions.

**Conventions:** short, **unsigned** commit messages; push to `origin` as work progresses. Docker build/run is verified by the user on a second machine (no daemon on the dev box) — verify Python natively + `docker compose config`.

---

## File Structure

| Path | Responsibility |
|------|----------------|
| `config.yaml` / `config.py` | Single source of truth + typed Pydantic loader (`log_target`, `ensemble` flags) |
| `data/features.py` | Parse raw string fields → engineered numeric/categorical features; drop invalid rows |
| `data/synthetic.py` | Deterministic synthetic generator (engineered schema) |
| `data/load.py` | Load real CSV (→ engineer) or synthetic + structural validation |
| `data/prepare.py` | Train/test split + preprocessing `ColumnTransformer` |
| `model/evaluate.py` | MAE/RMSE/R² |
| `model/train.py` | FLAML fit (ensemble + optional log-target via `TransformedTargetRegressor`) → artifact + metrics |
| `app/schemas.py` | Pydantic request/response models (enums) |
| `app/inference.py` | Shared payload→prediction helper (API + UI) |
| `app/api.py` | FastAPI `/predict`, `/health`, `/model-info` |
| `app/ui.py` | Streamlit form; API or standalone local model |
| infra | `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `packages.txt`, `Makefile`, `pyproject.toml`, `.pylintrc`, `.github/workflows/ci.yml` |
| `tests/` | pytest suite | `docs/` | data card, EDA, spec, this plan |

---

## Task 1: Scaffolding, config, dependencies, dataset

**Files:** `config.yaml`, `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`, `.pylintrc`, `packages.txt`, package `__init__.py`s, `.gitignore`, `data/raw/laptop_data.csv`.

- [ ] **Step 1: Packages + keep-files.** `data/__init__.py`, `model/__init__.py`, `app/__init__.py`, `tests/__init__.py` (each `"""Package marker."""`); `model/artifacts/.gitkeep`, `data/raw/.gitkeep`.

- [ ] **Step 2: `config.yaml`**

```yaml
data:
  raw_path: data/raw/laptop_data.csv
  synthetic: {enabled: true, n_rows: 1300, seed: 42}
  target: Price
  numeric_features: [Ram, Weight, Inches, ppi, SSD, HDD, Touchscreen, Ips, Cpu_rank]
  categorical_features: [Company, TypeName, Gpu_brand, Os]
  test_size: 0.2
  random_state: 42
model:
  task: regression
  time_budget_s: 60
  metric: r2
  estimator_list: [lgbm]
  ensemble: false
  log_target: true
  monotone_increasing: [Ram, SSD, HDD, ppi, Cpu_rank]
  artifact_dir: model/artifacts
  artifact_name: model.joblib
  metrics_name: metrics.json
  seed: 42
api: {host: 0.0.0.0, port: 8000}
ui: {api_url: "http://localhost:8000"}
```

- [ ] **Step 3: `requirements.txt`** (pinned to versions installed and verified on Python 3.13, which also have 3.11 wheels for the image): pandas, numpy, scikit-learn, flaml, lightgbm, joblib, PyYAML, pydantic, fastapi, uvicorn[standard], streamlit, requests. `requirements-dev.txt`: `-r requirements.txt` + pytest, pylint, black, isort, httpx, matplotlib.

- [ ] **Step 4: `pyproject.toml`** — black line-length 100; isort profile black; `[tool.pytest.ini_options] pythonpath=["."]`, `addopts="-q"`.

- [ ] **Step 5: `.pylintrc`** — `max-line-length=100`; `good-names=df,rng,ex,_,x_train,x_test,y_train,y_test`; `disable=too-few-public-methods,duplicate-code`.

- [ ] **Step 6: `packages.txt`** — `libgomp1` (LightGBM runtime on Streamlit Cloud).

- [ ] **Step 7: `.gitignore`** — append: ignore `data/raw/*` and `model/artifacts/*` except `!data/raw/laptop_data.csv`, `!data/raw/.gitkeep`, `!model/artifacts/.gitkeep`; ignore `.idea/`, `uv.lock`, `.playwright-mcp/`. Add `.gitattributes` (`* text=auto eol=lf`, `*.png binary`, `*.joblib binary`).

- [ ] **Step 8: Dataset.** Download the raw laptop CSV to `data/raw/laptop_data.csv` (campusx mirror) and commit it (small, public; makes clone → run work).

- [ ] **Step 9: Install + verify.** `pip install -r requirements-dev.txt`; `python -c "from flaml import AutoML; import lightgbm, fastapi, streamlit; print('ok')"`.

- [ ] **Step 10: Commit.** `git add … && git add -f data/raw/laptop_data.csv && git commit -m "scaffold project: config, deps, packages, dataset"`

---

## Task 2: `config.py` — typed config loader

**Files:** `config.py`; Test: `tests/test_config.py`.

- [ ] **Step 1: Failing test** — `load_config()` returns `AppConfig`; `cfg.data.target == "Price"`; `feature_columns == numeric + categorical`; `artifact_path.name == "model.joblib"`; missing file raises `FileNotFoundError`.

- [ ] **Step 2–4:** Implement Pydantic models. `ModelConfig` includes `ensemble: bool = True` and `log_target: bool = False`. `AppConfig` exposes `feature_columns`, `artifact_path`, `metrics_path` properties. `load_config(path="config.yaml")` reads YAML → `AppConfig` (raises `FileNotFoundError` if absent). Run test → pass.

- [ ] **Step 5: Commit** — `git commit -m "add config loader"`

---

## Task 3: `data/features.py` — feature engineering + cleaning

**Files:** `data/features.py`; Test: covered via `tests/test_load.py` (engineering runs inside `load_data`).

- [ ] **Step 1: Implement** (the novel, dataset-specific part)

```python
"""Feature engineering: turn the raw laptop CSV into clean model-ready columns."""
from __future__ import annotations

import pandas as pd

ENGINEERED_COLUMNS = [
    "Company", "TypeName", "Inches", "Ram", "Weight", "Touchscreen", "Ips",
    "ppi", "Cpu_rank", "SSD", "HDD", "Gpu_brand", "Os", "Price",
]


def _cpu_brand(text: str) -> str:
    head = " ".join(str(text).split()[:3])
    if head in {"Intel Core i7", "Intel Core i5", "Intel Core i3"}:
        return head
    return "Other Intel" if str(text).split()[0] == "Intel" else "AMD"


def _parse_memory(text: str) -> pd.Series:
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
    text = str(value)
    if "Windows" in text:
        return "Windows"
    if "Mac" in text or "macOS" in text:
        return "Mac"
    return "Other"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["Ram"] = pd.to_numeric(df["Ram"].astype(str).str.replace("GB", ""), errors="coerce")
    df["Weight"] = pd.to_numeric(df["Weight"].astype(str).str.replace("kg", ""), errors="coerce")
    df["Touchscreen"] = df["ScreenResolution"].str.contains("Touchscreen").astype(int)
    df["Ips"] = df["ScreenResolution"].str.contains("IPS").astype(int)
    resolution = df["ScreenResolution"].str.extract(r"(\d+)x(\d+)").astype(float)
    df["ppi"] = ((resolution[0] ** 2 + resolution[1] ** 2) ** 0.5 / df["Inches"]).round(2)
    df["Cpu_rank"] = df["Cpu"].apply(lambda t: CPU_RANK[_cpu_brand(t)])
    df[["SSD", "HDD"]] = df["Memory"].apply(_parse_memory)
    df["Gpu_brand"] = df["Gpu"].apply(lambda text: str(text).split()[0])
    df = df[df["Gpu_brand"] != "ARM"]
    df["Os"] = df["OpSys"].apply(_os_group)
    df = df.dropna(subset=["Ram", "Weight", "ppi"])
    return df[[c for c in ENGINEERED_COLUMNS if c in df.columns]]
```

- [ ] **Step 2: Sanity check** — `python -c "import pandas as pd; from data.features import engineer_features; print(engineer_features(pd.read_csv('data/raw/laptop_data.csv')).shape)"` → ~ (1302, 14).

- [ ] **Step 3: Commit** (with Task 4–5 as the data layer).

---

## Task 4: `data/synthetic.py` — synthetic generator

**Files:** `data/synthetic.py`; Test: `tests/test_synthetic.py`.

- [ ] **Step 1: Failing test** — `generate(50, 1).shape == (50, 14)`; `list(columns) == COLUMNS`; deterministic for same seed; injects nulls in `Weight`/`ppi`; `Price` is positive int; `Company`/`Price` never null.

- [ ] **Step 2: Implement** — `generate(n_rows=1300, seed=42)` builds the 13 engineered features (brand/type/CPU/GPU/OS categoricals, RAM/SSD/HDD/weight/inches/ppi/touch/IPS numerics) and a noisy linear `Price` (base + RAM/SSD/HDD/ppi terms + CPU/GPU/type/brand premiums + touch/IPS + Gaussian noise; clipped ≥ 9000; int). Inject ~2% nulls in `Weight`/`ppi`. Add `# pylint: disable=too-many-locals` (one var per column). Run test → pass.

- [ ] **Step 3: Commit** with the data layer.

---

## Task 5: `data/load.py` — load + validate

**Files:** `data/load.py`; Test: `tests/test_load.py`.

- [ ] **Step 1: Failing test** — real CSV present → `Price` in columns, non-empty; synthetic fallback (bogus `raw_path`) → `len == n_rows`; `validate_schema` raises on a missing column; allows per-cell nulls.

- [ ] **Step 2: Implement**

```python
def load_data(config: AppConfig) -> pd.DataFrame:
    raw_path = Path(config.data.raw_path)
    if raw_path.exists():
        df = engineer_features(pd.read_csv(raw_path))   # real -> engineer
    elif config.data.synthetic.enabled:
        df = synthetic.generate(config.data.synthetic.n_rows, config.data.synthetic.seed)
    else:
        raise FileNotFoundError(...)
    validate_schema(df, config, require_target=True)
    return df
```

`validate_schema` checks required columns (`feature_columns` + target) exist and the frame is non-empty; per-cell nulls allowed (imputed in prepare). Run test → pass.

- [ ] **Step 3: Commit data layer** — `git commit -m "pivot data layer to laptop dataset (FE + synthetic + config)"`

---

## Task 6: `data/prepare.py` — split + preprocessing

**Files:** `data/prepare.py`; Test: `tests/test_prepare.py`.

- [ ] **Step 1: Failing test** — split shapes sum to len; `x_train.columns == feature_columns`; preprocessor `fit_transform(train)` and `transform(test)` keep row counts (handle nulls + unseen categories).

- [ ] **Step 2: Implement** — `split_data` selects `feature_columns` + target, `train_test_split`. `build_preprocessor`: `ColumnTransformer([("num", SimpleImputer(median), numeric), ("cat", Pipeline([SimpleImputer(most_frequent), OneHotEncoder(handle_unknown="ignore", sparse_output=False)]), categorical)])` then `.set_output(transform="pandas")`. Run test → pass.

- [ ] **Step 3: Commit** with the data layer.

---

## Task 7: `model/evaluate.py` — metrics

**Files:** `model/evaluate.py`; Test: `tests/test_evaluate.py`.

- [ ] **Steps:** `regression_metrics(y_true, y_pred)` → `{mae, rmse, r2}` rounded (uses `mean_absolute_error`, `root_mean_squared_error`, `r2_score`). Test: perfect prediction → mae=rmse=0, r2=1; keys present + floats. Commit `-m "add regression metrics"`.

---

## Task 8: `model/train.py` — FLAML + log-target

**Files:** `model/train.py`; Test: `tests/test_train.py`.

- [ ] **Step 1: Failing test** — with a fast config (`time_budget_s=5`, tmp `artifact_dir`): `train(cfg)` writes `model.joblib` + `metrics.json`, `report["mae"] >= 0`, `report["r2"] <= 1.0`, `data_source` in {real, synthetic}, has `best_estimator`; the saved pipeline predicts `> 0` for a one-row laptop frame.

- [ ] **Step 2: Implement**

```python
def train(config: AppConfig) -> Dict:
    df = load_data(config)
    data_source = "real" if Path(config.data.raw_path).exists() else "synthetic"
    x_train, x_test, y_train, y_test = split_data(df, config)

    model_step = AutoML()
    if config.model.log_target:
        model_step = TransformedTargetRegressor(
            regressor=AutoML(), func=np.log1p, inverse_func=np.expm1
        )
    pipeline = Pipeline([("prep", build_preprocessor(config)), ("model", model_step)])
    pipeline.fit(
        x_train, y_train,
        model__task=config.model.task, model__time_budget=config.model.time_budget_s,
        model__metric=config.model.metric, model__estimator_list=config.model.estimator_list,
        model__ensemble=config.model.ensemble, model__seed=config.model.seed, model__verbose=0,
    )

    metrics = regression_metrics(y_test, pipeline.predict(x_test))
    fitted_model = pipeline.named_steps["model"]
    automl = getattr(fitted_model, "regressor_", fitted_model)  # unwrap TTR
    report = {**metrics, "best_estimator": automl.best_estimator, "training_date": ...,
              "n_rows": int(len(df)), "data_source": data_source,
              "feature_columns": config.feature_columns, "target": config.data.target,
              "feature_importance": _feature_importance(pipeline, x_test, y_test, config.model.seed)}
    # mkdir artifact_dir; joblib.dump(pipeline, artifact_path); json.dump(report, metrics_path)
    return report
```

`_feature_importance(pipeline, x_test, y_test, seed)` uses `sklearn.inspection.permutation_importance(scoring="r2")` keyed by `x_test.columns` (model-agnostic — works through the TTR + ensemble). `main()` runs `train(load_config())` with logging.

- [ ] **Step 3:** Run test → pass. Train the real model: `python -m model.train` (≈ R² 0.85, INR).

- [ ] **Step 4: Commit** — `git commit -m "pivot model: log-target (transformed target regressor)"`

---

## Task 9: `app/schemas.py` — Pydantic models

**Files:** `app/schemas.py`; Test: `tests/test_schemas.py`.

- [ ] **Steps:** Enums `Company` (19 brands), `TypeName`, `CpuBrand`, `GpuBrand`, `Os`. `PredictRequest` fields: `company, type_name, inches(>0), ram_gb(>=0), weight_kg(>0), touchscreen(0/1), ips(0/1), ppi(>0), cpu_brand, ssd_gb(>=0), hdd_gb(>=0), gpu_brand, os` with `examples`. `PredictResponse(price: float)`, `HealthResponse(status, model_loaded)`. Test: valid request; invalid enum → `ValidationError`; negative inches → `ValidationError`. Commit `-m "add api schemas"`.

---

## Task 10: `app/inference.py` — shared helper

**Files:** `app/inference.py`; Test: `tests/test_inference.py`.

- [ ] **Steps:** `REQUEST_TO_COLUMN` (snake_case field → engineered column); `to_feature_row(payload)` → one-row DataFrame; `predict_price(model, payload)` → `round(float(model.predict(...)[0]), 2)`. Test: `to_feature_row` shape (1,13) + 13 expected columns; `predict_price` > 0 with the trained model. Commit with the app layer.

---

## Task 11: `app/api.py` — FastAPI

**Files:** `app/api.py`; Test: `tests/test_api.py` + `tests/conftest.py` (`trained_model` fixture: train a fast model if the artifact is missing).

- [ ] **Steps:** `config = load_config()` at import; lazy cached `_load_model()`. `GET /health` → `HealthResponse`. `POST /predict` → 503 if no model, else `PredictResponse(price=predict_price(model, request.model_dump(mode="json")))`. `GET /model-info` → `metrics.json` (503 if absent). Test (TestClient): `/health` ok + model_loaded; `/predict` valid → 200, price > 0; invalid (`{"ram_gb":"abc"}`) → 422; `/model-info` has `r2`. Commit `-m "pivot app: laptop schemas, inference, api, ui"`.

---

## Task 12: `app/ui.py` — Streamlit (API or standalone)

**Files:** `app/ui.py`; Test: `tests/test_ui.py` (imports cleanly).

- [ ] **Steps:** module-level `API_URL`, `RESOLUTIONS`. `@st.cache_resource _local_model()` loads `model.joblib`, training once if missing. `get_prediction(payload)` tries the API, falls back to `predict_price(_local_model(), payload)`. `get_model_info()` tries the API, falls back to the local `metrics.json`. `main()` (`# pylint: disable=too-many-locals`): two-column form (brand/type/CPU/GPU/OS selectboxes from the schema enums; RAM/SSD/HDD selectboxes; weight/inches number inputs; resolution selectbox → compute PPI; touchscreen/IPS checkboxes), Predict button → `get_prediction`, plus a model-info expander with the feature-importance bar chart. Test: import sets `API_URL`. Commit with the app layer.

---

## Task 13: Docker + Compose

**Files:** `Dockerfile`, `.dockerignore`, `docker-compose.yml`.

- [ ] **Dockerfile:** `python:3.11-slim`; `apt-get install -y --no-install-recommends libgomp1`; copy `requirements.txt` first → `pip install`; copy app; `RUN python -m model.train` (bake the model); create non-root `appuser` + chown; `EXPOSE 8000 8501`; `CMD uvicorn app.api:app --host 0.0.0.0 --port 8000`.
- [ ] **docker-compose.yml:** services `api` (build, port 8000, healthcheck hitting `/health`) and `ui` (same image, `streamlit run app/ui.py`, `API_URL=http://api:8000`, port 8501, `depends_on: api: service_healthy`).
- [ ] **Validate:** `docker compose config` (syntax; build verified by the user on a second machine). Commit `-m "add docker, makefile, ci"` (with Task 14).

---

## Task 14: Makefile + CI

**Files:** `Makefile`, `.github/workflows/ci.yml`.

- [ ] **Makefile:** `install/train/api/ui/test/lint/format/docker` targets (+ documented `python -m …` equivalents).
- [ ] **CI:** checkout → setup-python 3.11 → `pip install -r requirements-dev.txt` → `pylint config.py data model app --fail-under=8` → `pytest`.
- [ ] Run `pylint config.py data model app` locally → ≥ 8 (target 10/10). Commit.

---

## Task 15: Docs — README + data card + EDA

**Files:** `docs/eda.py`, `docs/data_card.md`, `docs/img/*`, `README.md`.

- [ ] **`docs/eda.py`** (generic via config): histogram of `target`; scatter of `numeric_features[0]` (Ram) vs `target`; save `target_hist.png` + `feature_scatter.png`. (`# isort: skip_file` for the `sys.path` insert + late imports.)
- [ ] Generate plots + a feature-importance bar chart from `metrics.json`.
- [ ] **`data_card.md`** — source, raw→engineered table, cleaning, engineered schema, plots, baseline (R² ≈ 0.85).
- [ ] **`README.md`** — laptop summary, business context, architecture, repo tree, quickstart (Docker), native run, config (note `log_target`/`ensemble`), data + FE, retraining, API examples (laptop payload → `{price}`), testing/quality, **Deploy to Streamlit Cloud** (standalone UI + `packages.txt`), docs links. Embed the UI screenshot.
- [ ] Commit `-m "pivot docs + slides to laptop topic"`.

---

## Task 16: Proposal + slides

**Files:** `SUML_1.docx`, `slides.pptx`, `docs/build_slides.py`.

- [ ] **Proposal** (Polish, python-docx): fill title, app type, dataset (laptop + cleaning/FE), prediction (price, regression), ~150-word justification, ML model (FLAML ensemble + log-target, R² ≈ 0.85). Group index numbers = `[numery indeksów]` placeholder.
- [ ] **Slides** (`build_slides.py`, python-pptx, Polish, 10 slides): title; problem/value; data + cleaning (with scatter); architecture `data|model|app`; AutoML (FLAML + log-target, with feature-importance chart); results (R² 0,85 / MAE 9,6k / RMSE 14,6k); serving (FastAPI + Streamlit standalone); portability (Docker, one command); quality (pylint 10/10, 28 tests, CI); demo + summary. Validate via markitdown/text extraction. Commit `-m "add presentation slides"`.

---

## Task 17: Final verification + push

- [ ] `pytest` → all pass (28).
- [ ] `pylint config.py data model app --fail-under=8` → 10/10.
- [ ] `black --check . && isort --check-only .` → clean.
- [ ] End-to-end native smoke: boot `uvicorn app.api:app`, POST a laptop payload to `/predict` → positive price; `/model-info` → r2 ≈ 0.85; stop the server.
- [ ] `docker compose config` → valid.
- [ ] Capture a UI screenshot (start API + Streamlit; browser screenshot) → `docs/img/ui.png`.
- [ ] `git push -u origin main`.
- [ ] Hand the user the container checklist: `git clone … && docker compose up --build` → UI `:8501`, API `:8000/docs`.

---

## Self-Review

**Spec coverage:** data layer incl. FE/cleaning (Tasks 3–6) ✓; model + log-target + permutation importance (Tasks 7–8) ✓; API `/predict` `/health` `/model-info` + Pydantic (Tasks 9–11) ✓; Streamlit standalone UI (Task 12) ✓; config-driven incl. `log_target`/`ensemble` (Task 2) ✓; quality gates (Tasks 1,11,14,17) ✓; portability Docker/Makefile/packages.txt (Tasks 13–14) ✓; docs + deliverables (Tasks 15–16) ✓.

**Open dependency:** group index numbers (Task 16).

**Type consistency:** `AppConfig.feature_columns`/`artifact_path`/`metrics_path`/`log_target`/`ensemble`, `engineer_features`, `load_data`, `split_data`, `build_preprocessor`, `regression_metrics`, `train(config)->dict`, `REQUEST_TO_COLUMN`/`to_feature_row`/`predict_price`, and the Pydantic enums are used identically across tasks; engineered column names (`Ram`, `Cpu_rank`, `ppi`, …) match `features.py`, `config.yaml` and the schemas.
