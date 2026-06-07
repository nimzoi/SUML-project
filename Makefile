.PHONY: install train api ui mlflow test lint format validate docker

install:
	python -m pip install -r requirements-dev.txt

train:
	python -m model.train

api:
	uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run app/ui.py

mlflow:
	mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000

test:
	pytest

lint:
	pylint config.py data model app

format:
	isort . && black .

validate:
	python -c "from config import load_config; from model.retraining import validate_training_data; cfg=load_config(); report=validate_training_data(cfg); print(report.model_dump_json(indent=2)); raise SystemExit(0 if report.ok else 1)"

docker:
	docker compose up --build
