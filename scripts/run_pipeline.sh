#!/bin/bash
set -e

echo "Generating synthetic data..."
python -m src.data.generate_synthetic

echo "Training XGBoost model..."
python -m src.models.train_xgboost

echo "Running tests..."
pytest tests/ -v

echo "Pipeline completed successfully."