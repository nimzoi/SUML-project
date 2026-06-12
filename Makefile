.PHONY: install train api ui mlflow test lint format validate docker

# Wszystkie cele uzywaja lokalnego .venv, wiec dzialaja bez recznej aktywacji srodowiska.
# make uruchamia recepty przez sh, dlatego sciezka uzywa ukosnikow (dziala tez na Windows).
ifeq ($(OS),Windows_NT)
    VENV_PY := .venv/Scripts/python.exe
else
    VENV_PY := .venv/bin/python
endif

# Jeden krok: tworzy .venv i instaluje wszystko (runtime + dev). Odpowiednik setup.ps1.
install:
	python -m venv .venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements-dev.txt

train:
	$(VENV_PY) -m model.train

api:
	$(VENV_PY) -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

ui:
	$(VENV_PY) -m streamlit run app/ui.py

mlflow:
	$(VENV_PY) -m mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000

test:
	$(VENV_PY) -m pytest

lint:
	$(VENV_PY) -m pylint config.py data model app

format:
	$(VENV_PY) -m isort .
	$(VENV_PY) -m black .

validate:
	$(VENV_PY) -c "from config import load_config; from model.retraining import validate_training_data; cfg=load_config(); report=validate_training_data(cfg); print(report.model_dump_json(indent=2)); raise SystemExit(0 if report.ok else 1)"

docker:
	docker compose up --build
