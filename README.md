# Fistula Rehabilitation Platform – Phase One (Post-Surgical Triage)

AI‑enabled triage system for women recovering from obstetric fistula surgery in Sub‑Saharan Africa.

## Features

- **Synthetic data generation** with realistic noise, code‑switching, and rare dangerous presentations.
- **Two model pipelines:** XGBoost (fast baseline) and fine‑tuned DistilBERT (deep learning).
- **Cost‑sensitive training** prioritising urgent complications (sensitivity >95%).
- **Active learning** loop for uncertain predictions.
- **Embedding caching** for efficient inference.
- **MLflow experiment tracking**.
- **Dockerised** for reproducibility.

## Quick Start

```bash
# Clone and enter directory
git clone <repo>
cd fistula_phase1

# Set up virtual environment (optional)
python -m venv venv
source venv/bin/activate

# Install dependencies
make install

# Generate synthetic data and train model
make run-pipeline

# Start inference API (optional)
make api