.PHONY: help install dev-install pre-commit generate-data train-xgboost train-distilbert test run-pipeline clean api

help:
	@echo "Available commands:"
	@echo "  install          Install production dependencies"
	@echo "  dev-install      Install dev dependencies"
	@echo "  pre-commit       Install pre-commit hooks"
	@echo "  generate-data    Generate synthetic data"
	@echo "  train-xgboost    Train XGBoost model"
	@echo "  train-distilbert Train DistilBERT model"
	@echo "  test             Run tests"
	@echo "  run-pipeline     Full pipeline (generate + train + test)"
	@echo "  clean            Remove generated files"
	@echo "  api              Start FastAPI inference server"

install:
	pip install -r requirements.txt

dev-install: install
	pip install -r requirements-dev.txt
	pre-commit install

pre-commit:
	pre-commit run --all-files

generate-data:
	python -m src.data.generate_synthetic

train-xgboost:
	python -m src.models.train_xgboost

train-distilbert:
	python -m src.models.train_distilbert

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

run-pipeline: generate-data train-xgboost test

api:
	python scripts/start_api.py

clean:
	rm -rf data/fistula_rehab.db data/synthetic/*.csv data/processed/*.csv
	rm -rf models_artifacts/*.pkl models_artifacts/distilbert_finetuned models_artifacts/mlflow
	find . -type d -name __pycache__ -exec rm -rf {} +