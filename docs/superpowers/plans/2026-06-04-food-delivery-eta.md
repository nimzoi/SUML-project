# Food Delivery ETA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portable, reproducible regression app that predicts food-delivery time (minutes) via FLAML AutoML, served by FastAPI + Streamlit, containerized with Docker Compose.

**Architecture:** Strict `data | model | app` packages driven by a single `config.yaml`. `data` loads the real CSV (or generates synthetic data with the same schema) and validates/prepares it; `model` trains via FLAML and persists one `sklearn.Pipeline` artifact (preprocessor + model) plus `metrics.json`; `app` serves `/predict`, `/health`, `/model-info` over FastAPI with a Streamlit UI calling the API.

**Tech Stack:** Python 3.11+, pandas, scikit-learn, FLAML + LightGBM, FastAPI + uvicorn, Streamlit, Pydantic, PyYAML, joblib; pytest, pylint, black, isort; Docker Compose; GitHub Actions.

**Conventions for this project:** commit messages are short and **unsigned** (no Co-Authored-By trailer). Pushing to `origin` (github.com/nimzoi/SUML-project) as work progresses is pre-approved. Docker build/run is NOT verifiable on the dev machine (broken daemon) — verify Python natively + `docker compose config` only.

---

## File Structure

| Path | Responsibility |
|------|----------------|
| `config.yaml` | Single source of truth (data + model + serving settings) |
| `config.py` | Load + validate `config.yaml` into a typed Pydantic `AppConfig` |
| `data/synthetic.py` | Deterministic synthetic dataset matching the real schema |
| `data/load.py` | Load real CSV (or synthetic fallback) + structural validation |
| `data/prepare.py` | Train/test split + preprocessing `ColumnTransformer` |
| `model/evaluate.py` | Regression metrics (MAE/RMSE/R²) |
| `model/train.py` | FLAML fit → persist `model.joblib` + `metrics.json` |
| `app/schemas.py` | Pydantic request/response models (+ enums) |
| `app/api.py` | FastAPI: `/predict`, `/health`, `/model-info` |
| `app/ui.py` | Streamlit UI calling the API |
| `requirements.txt` | Pinned runtime deps (installed in Docker image) |
| `requirements-dev.txt` | Dev/test deps (pytest, pylint, black, isort, httpx, matplotlib) |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore` | Containerization |
| `Makefile`, `pyproject.toml`, `.pylintrc` | DX, formatting, lint config |
| `.github/workflows/ci.yml` | CI: pylint (≥8) + pytest |
| `tests/` | pytest suite (data/model/api/schemas/ui) |
| `docs/data_card.md` + `docs/eda.py` | Dataset description + EDA plots |
| `README.md` | Updated run/usage docs |
| `SUML_1.docx`, `slides.pptx` | Course deliverables |

---

## Task 1: Scaffolding, config.yaml, dependencies, gitignore, dataset

**Files:**
- Create: `config.yaml`, `requirements.txt`, `requirements-dev.txt`, `.pylintrc`, `pyproject.toml`
- Create: `data/__init__.py`, `model/__init__.py`, `app/__init__.py`, `tests/__init__.py`
- Create: `data/raw/.gitkeep`, `model/artifacts/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Create package markers and keep-files**

`data/__init__.py`, `model/__init__.py`, `app/__init__.py`, `tests/__init__.py` — each a single line:
```python
"""Package marker."""
```
Create empty `model/artifacts/.gitkeep` (the `data/raw/.gitkeep` too; the dataset already lives in `data/raw/`).

- [ ] **Step 2: Write `config.yaml`**

```yaml
data:
  raw_path: data/raw/Food_Delivery_Times.csv
  synthetic:
    enabled: true
    n_rows: 1000
    seed: 42
  target: Delivery_Time_min
  numeric_features:
    - Distance_km
    - Preparation_Time_min
    - Courier_Experience_yrs
  categorical_features:
    - Weather
    - Traffic_Level
    - Time_of_Day
    - Vehicle_Type
  test_size: 0.2
  random_state: 42
model:
  task: regression
  time_budget_s: 60
  metric: mae
  estimator_list:
    - lgbm
    - rf
    - extra_tree
  artifact_dir: model/artifacts
  artifact_name: model.joblib
  metrics_name: metrics.json
  seed: 42
api:
  host: 0.0.0.0
  port: 8000
ui:
  api_url: http://localhost:8000
```

- [ ] **Step 3: Write `requirements.txt` (runtime)**

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.2
flaml==2.3.4
lightgbm==4.5.0
joblib==1.4.2
PyYAML==6.0.2
pydantic==2.9.2
fastapi==0.115.6
uvicorn[standard]==0.34.0
streamlit==1.40.2
requests==2.32.3
```

`requirements-dev.txt`:
```
-r requirements.txt
pytest==8.3.4
pylint==3.3.3
black==24.10.0
isort==5.13.2
httpx==0.28.1
matplotlib==3.9.3
```

- [ ] **Step 4: Write `pyproject.toml`**

```toml
[tool.black]
line-length = 100
target-version = ["py311"]

[tool.isort]
profile = "black"
line_length = 100

[tool.pytest.ini_options]
pythonpath = ["."]
addopts = "-q"
```

- [ ] **Step 5: Write `.pylintrc`**

```ini
[FORMAT]
max-line-length=100

[BASIC]
good-names=df,rng,ex,_,x_train,x_test,y_train,y_test

[MESSAGES CONTROL]
disable=too-few-public-methods,duplicate-code
```

- [ ] **Step 6: Append data/artifact rules to `.gitignore`**

Append:
```
# Project data & artifacts (track the canonical dataset, ignore everything else)
data/raw/*
!data/raw/.gitkeep
!data/raw/Food_Delivery_Times.csv
model/artifacts/*
!model/artifacts/.gitkeep
```

- [ ] **Step 7: Install dependencies and verify imports**

Run: `python -m pip install -r requirements-dev.txt`
Then: `python -c "import flaml, lightgbm, fastapi, streamlit, sklearn, pandas; from flaml import AutoML; print('ok')"`
Expected: prints `ok`. If `from flaml import AutoML` fails, change `flaml==2.3.4` to `flaml[automl]==2.3.4` in both requirements files and reinstall. If a pinned version fails to resolve on Python 3.13, relax that single pin to the nearest installable version and record it.

- [ ] **Step 8: Commit**

```bash
git add config.yaml requirements.txt requirements-dev.txt pyproject.toml .pylintrc .gitignore \
        data/__init__.py model/__init__.py app/__init__.py tests/__init__.py \
        model/artifacts/.gitkeep
git add -f data/raw/Food_Delivery_Times.csv
git commit -m "scaffold project: config, deps, packages, dataset"
```

---

## Task 2: `config.py` — typed config loader

**Files:**
- Create: `config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the configuration loader."""
import pytest

from config import AppConfig, load_config


def test_load_config_returns_appconfig():
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.data.target == "Delivery_Time_min"
    assert cfg.model.task == "regression"


def test_feature_columns_combines_numeric_and_categorical():
    cfg = load_config()
    assert cfg.feature_columns == cfg.data.numeric_features + cfg.data.categorical_features


def test_artifact_and_metrics_paths():
    cfg = load_config()
    assert cfg.artifact_path.name == "model.joblib"
    assert cfg.metrics_path.name == "metrics.json"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist.yaml")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'config'` (or ImportError).

- [ ] **Step 3: Write `config.py`**

```python
"""Load and validate the project configuration from config.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import List, Union

import yaml
from pydantic import BaseModel, Field


class SyntheticConfig(BaseModel):
    """Settings for synthetic data generation."""

    enabled: bool = True
    n_rows: int = 1000
    seed: int = 42


class DataConfig(BaseModel):
    """Dataset location, schema and split settings."""

    raw_path: str
    synthetic: SyntheticConfig
    target: str
    numeric_features: List[str]
    categorical_features: List[str]
    test_size: float = 0.2
    random_state: int = 42


class ModelConfig(BaseModel):
    """AutoML and artifact settings."""

    task: str = "regression"
    time_budget_s: int = 60
    metric: str = "mae"
    estimator_list: List[str] = Field(default_factory=lambda: ["lgbm", "rf", "extra_tree"])
    artifact_dir: str = "model/artifacts"
    artifact_name: str = "model.joblib"
    metrics_name: str = "metrics.json"
    seed: int = 42


class ApiConfig(BaseModel):
    """FastAPI host/port."""

    host: str = "0.0.0.0"
    port: int = 8000


class UiConfig(BaseModel):
    """Streamlit settings."""

    api_url: str = "http://localhost:8000"


class AppConfig(BaseModel):
    """Top-level application configuration."""

    data: DataConfig
    model: ModelConfig
    api: ApiConfig
    ui: UiConfig

    @property
    def feature_columns(self) -> List[str]:
        """All model input columns (numeric first, then categorical)."""
        return self.data.numeric_features + self.data.categorical_features

    @property
    def artifact_path(self) -> Path:
        """Full path to the persisted model artifact."""
        return Path(self.model.artifact_dir) / self.model.artifact_name

    @property
    def metrics_path(self) -> Path:
        """Full path to the persisted metrics/metadata file."""
        return Path(self.model.artifact_dir) / self.model.metrics_name


def load_config(path: Union[str, Path] = "config.yaml") -> AppConfig:
    """Read config.yaml and return a validated AppConfig.

    Raises FileNotFoundError if the file is missing and pydantic.ValidationError
    if required keys are absent or malformed.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return AppConfig(**raw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "add config loader"
```

---

## Task 3: `data/synthetic.py` — synthetic data generator

**Files:**
- Create: `data/synthetic.py`
- Test: `tests/test_synthetic.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the synthetic data generator."""
from data.synthetic import COLUMNS, generate


def test_generate_shape_and_columns():
    df = generate(n_rows=50, seed=1)
    assert df.shape == (50, 9)
    assert list(df.columns) == COLUMNS


def test_generate_is_deterministic():
    assert generate(n_rows=20, seed=7).equals(generate(n_rows=20, seed=7))


def test_generate_injects_nulls():
    df = generate(n_rows=300, seed=3)
    assert df["Weather"].isnull().any()
    assert df["Courier_Experience_yrs"].isnull().any()
    # Target and Order_ID never null
    assert df["Delivery_Time_min"].notnull().all()
    assert df["Order_ID"].notnull().all()


def test_target_is_positive_integer():
    df = generate(n_rows=100, seed=2)
    assert (df["Delivery_Time_min"] > 0).all()
    assert str(df["Delivery_Time_min"].dtype).startswith("int")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthetic.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.synthetic'`.

- [ ] **Step 3: Write `data/synthetic.py`**

```python
"""Generate a synthetic Food Delivery dataset matching the real Kaggle schema."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

COLUMNS: List[str] = [
    "Order_ID",
    "Distance_km",
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
    "Delivery_Time_min",
]

WEATHER = ["Clear", "Rainy", "Snowy", "Foggy", "Windy"]
TRAFFIC = ["Low", "Medium", "High"]
TIME_OF_DAY = ["Morning", "Afternoon", "Evening", "Night"]
VEHICLE = ["Bike", "Scooter", "Car"]

_WEATHER_PENALTY = {"Clear": 0, "Windy": 3, "Foggy": 6, "Rainy": 8, "Snowy": 12}
_TRAFFIC_PENALTY = {"Low": 0, "Medium": 6, "High": 14}
_VEHICLE_SPEED = {"Bike": 2.5, "Scooter": 2.0, "Car": 1.6}
_NULL_COLUMNS = ["Weather", "Traffic_Level", "Time_of_Day", "Courier_Experience_yrs"]


def generate(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame with the same columns/types as the real dataset.

    Delivery time grows with distance, prep time, traffic and bad weather, and
    decreases slightly with courier experience, plus Gaussian noise. About 3% of
    cells in the same columns as the real data are set to NaN so the imputation
    path is exercised even without the real CSV.
    """
    rng = np.random.default_rng(seed)
    distance = np.round(rng.uniform(0.5, 20.0, n_rows), 2)
    prep = rng.integers(5, 30, n_rows)
    experience = np.round(rng.uniform(0.0, 10.0, n_rows), 1)
    weather = rng.choice(WEATHER, n_rows)
    traffic = rng.choice(TRAFFIC, n_rows)
    time_of_day = rng.choice(TIME_OF_DAY, n_rows)
    vehicle = rng.choice(VEHICLE, n_rows)

    minutes = (
        10.0
        + distance * np.array([_VEHICLE_SPEED[v] for v in vehicle])
        + prep * 0.8
        + np.array([_TRAFFIC_PENALTY[t] for t in traffic])
        + np.array([_WEATHER_PENALTY[w] for w in weather])
        - experience * 0.5
        + rng.normal(0.0, 5.0, n_rows)
    )
    minutes = np.clip(np.round(minutes), 5, None).astype(int)

    df = pd.DataFrame(
        {
            "Order_ID": np.arange(1, n_rows + 1),
            "Distance_km": distance,
            "Weather": weather,
            "Traffic_Level": traffic,
            "Time_of_Day": time_of_day,
            "Vehicle_Type": vehicle,
            "Preparation_Time_min": prep,
            "Courier_Experience_yrs": experience,
            "Delivery_Time_min": minutes,
        }
    )
    _inject_nulls(df, rng, frac=0.03)
    return df


def _inject_nulls(df: pd.DataFrame, rng: np.random.Generator, frac: float) -> None:
    """Set a fraction of cells to NaN in the columns that are nullable in real data."""
    n_null = int(len(df) * frac)
    for column in _NULL_COLUMNS:
        idx = rng.choice(len(df), size=n_null, replace=False)
        df.loc[idx, column] = np.nan
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthetic.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add data/synthetic.py tests/test_synthetic.py
git commit -m "add synthetic data generator"
```

---

## Task 4: `data/load.py` — load + validate

**Files:**
- Create: `data/load.py`
- Test: `tests/test_load.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for dataset loading and validation."""
import pandas as pd
import pytest

from config import load_config
from data.load import load_data, validate_schema


def test_load_real_csv_when_present():
    cfg = load_config()  # data/raw/Food_Delivery_Times.csv exists in the repo
    df = load_data(cfg)
    assert cfg.data.target in df.columns
    assert len(df) > 0


def test_synthetic_fallback_when_csv_missing():
    cfg = load_config()
    cfg.data.raw_path = "data/raw/__missing__.csv"
    df = load_data(cfg)
    assert len(df) == cfg.data.synthetic.n_rows


def test_validate_schema_raises_on_missing_column():
    cfg = load_config()
    with pytest.raises(ValueError):
        validate_schema(pd.DataFrame({"foo": [1, 2]}), cfg)


def test_validate_schema_allows_nulls():
    cfg = load_config()
    df = load_data(cfg)
    df.loc[df.index[:5], cfg.data.categorical_features[0]] = None
    # Should not raise: per-cell nulls are allowed
    validate_schema(df, cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_load.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.load'`.

- [ ] **Step 3: Write `data/load.py`**

```python
"""Load the dataset (real CSV if present, otherwise synthetic) and validate it."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import AppConfig
from data import synthetic

logger = logging.getLogger(__name__)


def load_data(config: AppConfig) -> pd.DataFrame:
    """Return the raw dataframe from the real CSV if it exists, else synthetic data."""
    raw_path = Path(config.data.raw_path)
    if raw_path.exists():
        logger.info("Loading real dataset from %s", raw_path)
        df = pd.read_csv(raw_path)
    elif config.data.synthetic.enabled:
        logger.info("Real CSV not found at %s; generating synthetic dataset", raw_path)
        df = synthetic.generate(config.data.synthetic.n_rows, config.data.synthetic.seed)
    else:
        raise FileNotFoundError(
            f"No dataset at {raw_path} and synthetic generation is disabled"
        )
    validate_schema(df, config, require_target=True)
    return df


def validate_schema(df: pd.DataFrame, config: AppConfig, require_target: bool = True) -> None:
    """Validate that required columns exist and the frame is non-empty.

    Per-cell nulls are allowed on purpose: the real data contains them and they are
    imputed later in `prepare`. Missing/extra columns raise a clear error to protect
    future retraining batches from silent corruption.
    """
    required = list(config.feature_columns)
    if require_target:
        required.append(config.data.target)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_load.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add data/load.py tests/test_load.py
git commit -m "add data loader with schema validation"
```

---

## Task 5: `data/prepare.py` — split + preprocessing

**Files:**
- Create: `data/prepare.py`
- Test: `tests/test_prepare.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for split and preprocessing."""
from config import load_config
from data.load import load_data
from data.prepare import build_preprocessor, split_data


def test_split_shapes_and_columns():
    cfg = load_config()
    df = load_data(cfg)
    x_train, x_test, y_train, y_test = split_data(df, cfg)
    assert len(x_train) + len(x_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)
    assert list(x_train.columns) == cfg.feature_columns


def test_preprocessor_handles_nulls_and_unknown_categories():
    cfg = load_config()
    df = load_data(cfg)
    x_train, x_test, _, _ = split_data(df, cfg)
    pre = build_preprocessor(cfg)
    transformed_train = pre.fit_transform(x_train)
    assert transformed_train.shape[0] == len(x_train)
    # Transforming unseen rows (possibly with nulls) must not raise
    transformed_test = pre.transform(x_test)
    assert transformed_test.shape[0] == len(x_test)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prepare.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.prepare'`.

- [ ] **Step 3: Write `data/prepare.py`**

```python
"""Split the data and build the preprocessing ColumnTransformer."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import AppConfig


def split_data(
    df: pd.DataFrame, config: AppConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train/test feature frames and target series."""
    features = df[config.feature_columns]
    target = df[config.data.target]
    return train_test_split(
        features,
        target,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
    )


def build_preprocessor(config: AppConfig) -> ColumnTransformer:
    """Impute + one-hot categoricals; impute numerics (no scaling for tree models)."""
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = SimpleImputer(strategy="median")
    return ColumnTransformer(
        transformers=[
            ("num", numeric, config.data.numeric_features),
            ("cat", categorical, config.data.categorical_features),
        ]
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prepare.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add data/prepare.py tests/test_prepare.py
git commit -m "add split and preprocessing"
```

---

## Task 6: `model/evaluate.py` — metrics

**Files:**
- Create: `model/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for regression metrics."""
from model.evaluate import regression_metrics


def test_perfect_prediction():
    metrics = regression_metrics([1, 2, 3], [1, 2, 3])
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_keys_present_and_floats():
    metrics = regression_metrics([1, 2, 3, 4], [1.5, 2.5, 2.0, 4.0])
    assert set(metrics) == {"mae", "rmse", "r2"}
    assert all(isinstance(v, float) for v in metrics.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.evaluate'`.

- [ ] **Step 3: Write `model/evaluate.py`**

```python
"""Regression metrics for model evaluation."""
from __future__ import annotations

from typing import Dict, Sequence

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """Return MAE, RMSE and R2 as a dict of rounded floats."""
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }
```

> Note: `root_mean_squared_error` requires scikit-learn ≥ 1.4. If the pinned version is older, compute RMSE as `mean_squared_error(y_true, y_pred) ** 0.5`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluate.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add model/evaluate.py tests/test_evaluate.py
git commit -m "add regression metrics"
```

---

## Task 7: `model/train.py` — FLAML training + artifact

**Files:**
- Create: `model/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the training pipeline."""
import joblib
import pandas as pd

from config import load_config
from model.train import train


def _fast_config(tmp_path):
    cfg = load_config()
    cfg.model.time_budget_s = 5
    cfg.model.artifact_dir = str(tmp_path)
    return cfg


def test_train_creates_artifact_and_metrics(tmp_path):
    cfg = _fast_config(tmp_path)
    report = train(cfg)
    assert cfg.artifact_path.exists()
    assert cfg.metrics_path.exists()
    assert report["mae"] >= 0
    assert report["data_source"] in {"real", "synthetic"}
    assert "best_estimator" in report


def test_saved_pipeline_predicts(tmp_path):
    cfg = _fast_config(tmp_path)
    train(cfg)
    pipeline = joblib.load(cfg.artifact_path)
    row = pd.DataFrame(
        [
            {
                "Distance_km": 7.9,
                "Preparation_Time_min": 12,
                "Courier_Experience_yrs": 2.0,
                "Weather": "Clear",
                "Traffic_Level": "Medium",
                "Time_of_Day": "Afternoon",
                "Vehicle_Type": "Scooter",
            }
        ]
    )
    prediction = pipeline.predict(row)
    assert prediction[0] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.train'`.

- [ ] **Step 3: Write `model/train.py`**

```python
"""Train the model with FLAML AutoML and persist the artifact + metrics."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import joblib
from flaml import AutoML
from sklearn.pipeline import Pipeline

from config import AppConfig, load_config
from data.load import load_data
from data.prepare import build_preprocessor, split_data
from model.evaluate import regression_metrics

logger = logging.getLogger(__name__)


def train(config: AppConfig) -> Dict:
    """Run the full training pipeline and write model.joblib + metrics.json."""
    df = load_data(config)
    data_source = "real" if Path(config.data.raw_path).exists() else "synthetic"
    x_train, x_test, y_train, y_test = split_data(df, config)

    pipeline = Pipeline(
        steps=[("prep", build_preprocessor(config)), ("model", AutoML())]
    )
    pipeline.fit(
        x_train,
        y_train,
        model__task=config.model.task,
        model__time_budget=config.model.time_budget_s,
        model__metric=config.model.metric,
        model__estimator_list=config.model.estimator_list,
        model__seed=config.model.seed,
        model__verbose=0,
    )

    metrics = regression_metrics(y_test, pipeline.predict(x_test))
    report = {
        **metrics,
        "best_estimator": pipeline.named_steps["model"].best_estimator,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df)),
        "data_source": data_source,
        "feature_columns": config.feature_columns,
        "target": config.data.target,
        "feature_importance": _feature_importance(pipeline),
    }

    Path(config.model.artifact_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.artifact_path)
    with config.metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Saved model to %s | metrics=%s", config.artifact_path, metrics)
    return report


def _feature_importance(pipeline: Pipeline, top_n: int = 15) -> Dict[str, float]:
    """Best-effort feature importance keyed by transformed feature name."""
    try:
        names = pipeline.named_steps["prep"].get_feature_names_out()
        estimator = pipeline.named_steps["model"].model.estimator
        importances = getattr(estimator, "feature_importances_", None)
        if importances is None:
            return {}
        ranked = sorted(zip(names, importances), key=lambda pair: pair[1], reverse=True)
        return {name: round(float(value), 4) for name, value in ranked[:top_n]}
    except (AttributeError, KeyError, ValueError):
        return {}


def main() -> None:
    """CLI entry point: train with config.yaml and log the result."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    train(load_config())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train.py -q`
Expected: 2 passed (each trains a 5s model — allow ~20s total).

- [ ] **Step 5: Train the real model for the rest of the build**

Run: `python -m model.train`
Expected: logs `Saved model to model/artifacts/model.joblib | metrics={...}`; `model/artifacts/model.joblib` and `metrics.json` now exist (git-ignored).

- [ ] **Step 6: Commit**

```bash
git add model/train.py tests/test_train.py
git commit -m "add flaml training and artifact persistence"
```

---

## Task 8: `app/schemas.py` — Pydantic API models

**Files:**
- Create: `app/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the API request/response schemas."""
import pytest
from pydantic import ValidationError

from app.schemas import PredictRequest


def _valid_payload():
    return {
        "distance_km": 7.9,
        "weather": "Clear",
        "traffic_level": "Medium",
        "time_of_day": "Afternoon",
        "vehicle_type": "Scooter",
        "preparation_time_min": 12,
        "courier_experience_yrs": 2.0,
    }


def test_valid_request():
    req = PredictRequest(**_valid_payload())
    assert req.distance_km == 7.9
    assert req.weather.value == "Clear"


def test_invalid_enum_value():
    payload = _valid_payload()
    payload["weather"] = "Sunny"
    with pytest.raises(ValidationError):
        PredictRequest(**payload)


def test_negative_distance_rejected():
    payload = _valid_payload()
    payload["distance_km"] = -1
    with pytest.raises(ValidationError):
        PredictRequest(**payload)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.schemas'`.

- [ ] **Step 3: Write `app/schemas.py`**

```python
"""Pydantic request/response models for the prediction API."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Weather(str, Enum):
    """Allowed weather categories."""

    CLEAR = "Clear"
    RAINY = "Rainy"
    SNOWY = "Snowy"
    FOGGY = "Foggy"
    WINDY = "Windy"


class TrafficLevel(str, Enum):
    """Allowed traffic levels."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class TimeOfDay(str, Enum):
    """Allowed times of day."""

    MORNING = "Morning"
    AFTERNOON = "Afternoon"
    EVENING = "Evening"
    NIGHT = "Night"


class VehicleType(str, Enum):
    """Allowed vehicle types."""

    BIKE = "Bike"
    SCOOTER = "Scooter"
    CAR = "Car"


class PredictRequest(BaseModel):
    """Input features for a single delivery-time prediction."""

    distance_km: float = Field(..., ge=0, examples=[7.9])
    weather: Weather = Field(..., examples=["Clear"])
    traffic_level: TrafficLevel = Field(..., examples=["Medium"])
    time_of_day: TimeOfDay = Field(..., examples=["Afternoon"])
    vehicle_type: VehicleType = Field(..., examples=["Scooter"])
    preparation_time_min: int = Field(..., ge=0, examples=[12])
    courier_experience_yrs: float = Field(..., ge=0, examples=[2.0])


class PredictResponse(BaseModel):
    """Predicted delivery time."""

    eta_minutes: float


class HealthResponse(BaseModel):
    """Liveness and model-loaded status."""

    status: str
    model_loaded: bool
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add app/schemas.py tests/test_schemas.py
git commit -m "add api schemas"
```

---

## Task 9: `app/api.py` — FastAPI service

**Files:**
- Create: `app/api.py`
- Test: `tests/conftest.py`, `tests/test_api.py`

- [ ] **Step 1: Write `tests/conftest.py` (shared fixtures)**

```python
"""Shared pytest fixtures."""
import pytest

from config import load_config
from model.train import train


@pytest.fixture(scope="session")
def trained_model():
    """Ensure a model artifact exists at the default path (fast budget if missing)."""
    cfg = load_config()
    if not cfg.artifact_path.exists():
        cfg.model.time_budget_s = 5
        train(cfg)
    return cfg
```

- [ ] **Step 2: Write the failing test**

```python
"""Tests for the FastAPI service."""
from fastapi.testclient import TestClient


def _client():
    from app.api import app

    return TestClient(app)


def _valid_payload():
    return {
        "distance_km": 7.9,
        "weather": "Clear",
        "traffic_level": "Medium",
        "time_of_day": "Afternoon",
        "vehicle_type": "Scooter",
        "preparation_time_min": 12,
        "courier_experience_yrs": 2.0,
    }


def test_health_ok(trained_model):
    response = _client().get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_eta(trained_model):
    response = _client().post("/predict", json=_valid_payload())
    assert response.status_code == 200
    assert response.json()["eta_minutes"] > 0


def test_predict_rejects_invalid(trained_model):
    response = _client().post("/predict", json={"distance_km": "abc"})
    assert response.status_code == 422


def test_model_info(trained_model):
    response = _client().get("/model-info")
    assert response.status_code == 200
    assert "mae" in response.json()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_api.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api'`.

- [ ] **Step 4: Write `app/api.py`**

```python
"""FastAPI service exposing the delivery-time model."""
from __future__ import annotations

import json
import logging
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import HealthResponse, PredictRequest, PredictResponse
from config import load_config

logger = logging.getLogger(__name__)
config = load_config()

app = FastAPI(title="Food Delivery ETA API", version="1.0.0")

# Map snake_case request fields to the model's original column names.
_REQUEST_TO_COLUMN = {
    "distance_km": "Distance_km",
    "weather": "Weather",
    "traffic_level": "Traffic_Level",
    "time_of_day": "Time_of_Day",
    "vehicle_type": "Vehicle_Type",
    "preparation_time_min": "Preparation_Time_min",
    "courier_experience_yrs": "Courier_Experience_yrs",
}

_cache: Dict[str, object] = {}


def _load_model():
    """Lazily load and cache the model artifact (None if not trained yet)."""
    if "model" not in _cache and config.artifact_path.exists():
        _cache["model"] = joblib.load(config.artifact_path)
        logger.info("Loaded model from %s", config.artifact_path)
    return _cache.get("model")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe; reports whether a model is loaded."""
    return HealthResponse(status="ok", model_loaded=_load_model() is not None)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict delivery time (minutes) for a single order."""
    model = _load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available. Train it first.")
    row = {
        column: getattr(request, field).value
        if hasattr(getattr(request, field), "value")
        else getattr(request, field)
        for field, column in _REQUEST_TO_COLUMN.items()
    }
    prediction = float(model.predict(pd.DataFrame([row]))[0])
    return PredictResponse(eta_minutes=round(prediction, 1))


@app.get("/model-info")
def model_info() -> Dict:
    """Return the persisted metrics/metadata for the current model."""
    if not config.metrics_path.exists():
        raise HTTPException(status_code=503, detail="No model metrics found. Train first.")
    with config.metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_api.py -q`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add app/api.py tests/conftest.py tests/test_api.py
git commit -m "add fastapi service"
```

---

## Task 10: `app/ui.py` — Streamlit UI

**Files:**
- Create: `app/ui.py`
- Test: `tests/test_ui.py`

- [ ] **Step 1: Write the failing test**

```python
"""Smoke test: the Streamlit script imports without error."""
import importlib


def test_ui_imports(monkeypatch):
    monkeypatch.setenv("API_URL", "http://localhost:8000")
    module = importlib.import_module("app.ui")
    importlib.reload(module)
    assert module.API_URL == "http://localhost:8000"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ui.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ui'`.

- [ ] **Step 3: Write `app/ui.py`**

```python
"""Streamlit UI for the Food Delivery ETA service."""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 10


def main() -> None:
    """Render the prediction form and call the API."""
    st.set_page_config(page_title="Food Delivery ETA", page_icon="🛵")
    st.title("🛵 Food Delivery ETA")
    st.caption("Estimate delivery time from order and route features.")

    distance = st.number_input("Distance (km)", min_value=0.0, value=7.9, step=0.1)
    weather = st.selectbox("Weather", ["Clear", "Rainy", "Snowy", "Foggy", "Windy"])
    traffic = st.selectbox("Traffic level", ["Low", "Medium", "High"])
    time_of_day = st.selectbox("Time of day", ["Morning", "Afternoon", "Evening", "Night"])
    vehicle = st.selectbox("Vehicle type", ["Bike", "Scooter", "Car"])
    prep = st.number_input("Preparation time (min)", min_value=0, value=12, step=1)
    experience = st.number_input("Courier experience (yrs)", min_value=0.0, value=2.0, step=0.5)

    if st.button("Predict ETA"):
        payload = {
            "distance_km": distance,
            "weather": weather,
            "traffic_level": traffic,
            "time_of_day": time_of_day,
            "vehicle_type": vehicle,
            "preparation_time_min": prep,
            "courier_experience_yrs": experience,
        }
        try:
            response = requests.post(
                f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            st.success(f"Estimated delivery time: {response.json()['eta_minutes']} min")
        except requests.RequestException as ex:
            st.error(f"Prediction failed: {ex}")

    with st.expander("Model info & feature importance"):
        try:
            info = requests.get(f"{API_URL}/model-info", timeout=REQUEST_TIMEOUT).json()
            st.write(
                {k: info[k] for k in ("mae", "rmse", "r2", "best_estimator") if k in info}
            )
            if info.get("feature_importance"):
                st.bar_chart(info["feature_importance"])
        except requests.RequestException as ex:
            st.warning(f"Could not load model info: {ex}")


if __name__ == "__main__":
    main()
```

> Note: the smoke test only imports the module (top-level code defines `API_URL` and `main`). Streamlit widgets run inside `main()`, so importing does not require a Streamlit runtime.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ui.py -q`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add app/ui.py tests/test_ui.py
git commit -m "add streamlit ui"
```

---

## Task 11: Docker (Dockerfile, .dockerignore, compose)

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `docker-compose.yml`

- [ ] **Step 1: Write `.dockerignore`**

```
.git
.idea
.github
docs
tests
__pycache__
*.pyc
.pytest_cache
.venv
venv
*.pdf
*.docx
*.pptx
```

- [ ] **Step 2: Write `Dockerfile`**

```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# libgomp1 is required by lightgbm at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train at build time so the image ships ready to serve.
RUN python -m model.train

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Write `docker-compose.yml`**

```yaml
services:
  api:
    build: .
    image: food-delivery-eta:latest
    command: uvicorn app.api:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"]
      interval: 30s
      timeout: 5s
      retries: 3
  ui:
    image: food-delivery-eta:latest
    command: streamlit run app/ui.py --server.port 8501 --server.address 0.0.0.0
    environment:
      - API_URL=http://api:8000
    ports:
      - "8501:8501"
    depends_on:
      api:
        condition: service_healthy
```

- [ ] **Step 4: Validate compose syntax (no daemon needed)**

Run: `docker compose config`
Expected: prints the normalized config with no error. (Image build/run is verified by the user on a second machine — do not claim it was tested here.)

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .dockerignore docker-compose.yml
git commit -m "add docker image and compose"
```

---

## Task 12: Makefile + CI

**Files:**
- Create: `Makefile`, `.github/workflows/ci.yml`

- [ ] **Step 1: Write `Makefile`**

```makefile
.PHONY: install train api ui test lint format docker

install:
	python -m pip install -r requirements-dev.txt

train:
	python -m model.train

api:
	uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run app/ui.py

test:
	pytest

lint:
	pylint config.py data model app

format:
	isort . && black .

docker:
	docker compose up --build
```

- [ ] **Step 2: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: python -m pip install -r requirements-dev.txt
      - name: Lint (pylint >= 8)
        run: pylint config.py data model app --fail-under=8
      - name: Tests
        run: pytest
```

- [ ] **Step 3: Run lint locally to confirm ≥ 8**

Run: `pylint config.py data model app --fail-under=8`
Expected: score ≥ 8.00/10 and exit code 0. Fix any warnings that drop it below 8 (add docstrings, remove unused imports, shorten long lines) before committing.

- [ ] **Step 4: Commit**

```bash
git add Makefile .github/workflows/ci.yml
git commit -m "add makefile and ci"
```

---

## Task 13: Docs — README + data card + EDA plots

**Files:**
- Create: `docs/eda.py`, `docs/data_card.md`, `docs/img/` (generated plots)
- Modify: `README.md`

- [ ] **Step 1: Write `docs/eda.py` (generates plots for the data card)**

```python
"""Generate EDA plots for the data card. Run: python docs/eda.py"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position

from config import load_config  # noqa: E402  pylint: disable=wrong-import-position
from data.load import load_data  # noqa: E402  pylint: disable=wrong-import-position

OUT = Path("docs/img")


def main() -> None:
    """Save a target histogram and a distance-vs-time scatter."""
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    df = load_data(cfg)

    plt.figure()
    df[cfg.data.target].hist(bins=30)
    plt.xlabel("Delivery time (min)")
    plt.ylabel("Count")
    plt.title("Delivery time distribution")
    plt.tight_layout()
    plt.savefig(OUT / "target_hist.png")

    plt.figure()
    plt.scatter(df["Distance_km"], df[cfg.data.target], s=8, alpha=0.4)
    plt.xlabel("Distance (km)")
    plt.ylabel("Delivery time (min)")
    plt.title("Distance vs delivery time")
    plt.tight_layout()
    plt.savefig(OUT / "distance_vs_time.png")
    print(f"Saved plots to {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the plots**

Run: `python docs/eda.py`
Expected: `Saved plots to docs/img`; two PNGs created.

- [ ] **Step 3: Write `docs/data_card.md`**

Content: dataset name/source (Kaggle `denkuznetz/food-delivery-time-prediction`), 1000×9 schema table (reuse the spec's table), null counts (30 each in Weather/Traffic_Level/Time_of_Day/Courier_Experience_yrs), target stats (min 8 / mean ~57 / max 153), and embed `![](img/target_hist.png)` and `![](img/distance_vs_time.png)`. Note that when the CSV is absent the synthetic generator reproduces this schema.

- [ ] **Step 4: Rewrite `README.md`**

Replace the `{{placeholder}}` template with concrete content: project summary; architecture diagram (`data | model | app`); **Quickstart (Docker):** `git clone ... && docker compose up --build` → API `http://localhost:8000/docs`, UI `http://localhost:8501`; **Run natively:** `pip install -r requirements-dev.txt`, `python -m model.train`, `make api`, `make ui`; **Config** section (point at `config.yaml`); **Retraining:** drop a new CSV in `data/raw/` and re-run `python -m model.train`; **API** examples for `/predict`, `/health`, `/model-info`; **Testing & quality:** `make test`, `make lint` (pylint ≥ 8); link to `docs/data_card.md` and the spec.

- [ ] **Step 5: Commit**

```bash
git add docs/eda.py docs/data_card.md docs/img README.md
git commit -m "add data card, eda plots, and rewrite readme"
```

---

## Task 14: SUML_1.docx — project proposal

**Files:**
- Modify: `SUML_1.docx`

> **Blocked on input:** needs the group's **index numbers**. Ask the user before this task. Title language: Polish.

- [ ] **Step 1: Read the current template structure**

Run: `python -c "import docx2txt; print(docx2txt.process('SUML_1.docx'))"`
(Confirm the field labels: index numbers, working title, app type, dataset, prediction, justification ~150 words, ML model.)

- [ ] **Step 2: Fill the proposal**

Use `python-docx` to fill each field (Polish):
- Index numbers: `<from user>`
- Tytuł roboczy: „Food Delivery ETA — predykcja czasu dostawy"
- Rodzaj aplikacji: web + API (Streamlit + FastAPI)
- Dataset: Kaggle `denkuznetz/food-delivery-time-prediction` (1000×9), z generatorem syntetycznym o tym samym schemacie; pobranie z Kaggle, w repo dołączony CSV
- Predykcja: czas dostawy w minutach (regresja)
- Uzasadnienie (~150 słów): wartość biznesowa ETA dla platform dostawczych (oczekiwania klienta, dyspozycja kurierów, flagowanie spóźnień)
- Model ML: AutoML (FLAML), regresja, model trenowany na potrzeby aplikacji; estymatory lgbm/rf/extra_tree

- [ ] **Step 3: Validate the document**

Run: `python -c "import docx; docx.Document('SUML_1.docx'); print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add SUML_1.docx
git commit -m "fill project proposal"
```

---

## Task 15: Presentation slides (.pptx)

**Files:**
- Create: `slides.pptx`

> Language: Polish (assumed). ~10-minute talk.

- [ ] **Step 1: Build the deck with python-pptx**

Slides: (1) Tytuł + zespół; (2) Problem & wartość biznesowa (ETA); (3) Dane (schemat, braki, syntetyk); (4) Architektura `data | model | app`; (5) AutoML (FLAML) + wyniki (MAE/RMSE/R² z `metrics.json`); (6) Wystawienie: FastAPI `/docs` + Streamlit; (7) Przenoszalność: Docker Compose, one-command; (8) Jakość: pylint ≥ 8, testy, CI; (9) Demo (skrypt: `docker compose up`, predykcja w UI); (10) Retraining + podsumowanie.

- [ ] **Step 2: Validate**

Run: `python -c "from pptx import Presentation; Presentation('slides.pptx'); print('ok')"`
Expected: `ok`. (Add `python-pptx` to dev deps if missing.)

- [ ] **Step 3: Commit**

```bash
git add slides.pptx
git commit -m "add presentation slides"
```

---

## Task 16: Final verification + push

- [ ] **Step 1: Full test suite**

Run: `pytest`
Expected: all tests pass.

- [ ] **Step 2: Lint gate**

Run: `pylint config.py data model app --fail-under=8`
Expected: ≥ 8.00/10, exit 0.

- [ ] **Step 3: Format check**

Run: `black --check . && isort --check-only .`
Expected: no changes needed (run `make format` first if it reports diffs).

- [ ] **Step 4: End-to-end native smoke**

Run (background): `uvicorn app.api:app --port 8000` then
`python -c "import requests,json; print(requests.post('http://localhost:8000/predict', json={'distance_km':7.9,'weather':'Clear','traffic_level':'Medium','time_of_day':'Afternoon','vehicle_type':'Scooter','preparation_time_min':12,'courier_experience_yrs':2.0}).json())"`
Expected: `{'eta_minutes': <positive float>}`. Stop the server afterward.

- [ ] **Step 5: Compose syntax**

Run: `docker compose config`
Expected: valid normalized config.

- [ ] **Step 6: Push**

```bash
git push -u origin main
```

- [ ] **Step 7: Hand the user the container test checklist**

On the second machine: `git clone https://github.com/nimzoi/SUML-project.git && cd SUML-project && docker compose up --build`, then open `http://localhost:8501` (UI) and `http://localhost:8000/docs` (API), submit a prediction, check `/model-info`.

---

## Self-Review

**Spec coverage:** data layer (Tasks 3–5) ✓; model + AutoML + artifacts/metrics (Tasks 6–7) ✓; API `/predict` `/health` `/model-info` + Pydantic (Tasks 8–9) ✓; Streamlit UI + feature importance (Task 10) ✓; config-driven (Task 2) ✓; quality gates CI/tests/format (Tasks 1,9,12,16) ✓; portability Docker/Makefile/.dockerignore (Tasks 11–12) ✓; ops/retraining metrics.json + logging + validation (Tasks 2,4,7,9) ✓; presentation aids data card + feature importance (Tasks 10,13) ✓; deliverables proposal + slides + README (Tasks 13–15) ✓; hybrid dataset + retraining-ready (Tasks 3–5,7) ✓.

**Open dependency:** Task 14 needs the group's index numbers (collect before starting it).

**Type consistency:** `AppConfig.feature_columns`/`artifact_path`/`metrics_path`, `load_data(config)`, `split_data`, `build_preprocessor`, `regression_metrics`, `train(config)->dict`, `_REQUEST_TO_COLUMN`, and the Pydantic enums are used identically across tasks. Column names (`Distance_km`, `Weather`, …) match the real CSV.
