.PHONY: setup lint test train serve mlflow docker-build

PYTHON ?= python3.11
VENV := .venv
BIN := $(VENV)/bin
PYTHONPATH_ENV := PYTHONPATH=.

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip setuptools wheel
	$(BIN)/pip install -r requirements.txt

lint:
	$(BIN)/ruff check .

test:
	$(PYTHONPATH_ENV) $(BIN)/pytest -q

train:
	$(PYTHONPATH_ENV) $(BIN)/python -m src.training.train --config configs/train.yaml --model_family ridge

serve:
	$(PYTHONPATH_ENV) $(BIN)/uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

mlflow:
	$(BIN)/mlflow ui --host 0.0.0.0 --port 5001

docker-build:
	docker build -f docker/Dockerfile -t stock-price-predictor:local .
