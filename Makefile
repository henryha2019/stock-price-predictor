.PHONY: setup lint test train serve docker-build

PYTHON := python3.11
VENV := .venv
BIN := $(VENV)/bin

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip setuptools wheel
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -e .

lint:
	$(BIN)/ruff check .

test:
	PYTHONPATH=. $(BIN)/pytest -q

train:
	$(BIN)/python -m src.training.train --config configs/train.yaml

serve:
	$(BIN)/uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -f docker/Dockerfile -t stock-price-predictor:local .
