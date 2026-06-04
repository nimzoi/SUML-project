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
