FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV DB_PATH=/app/data/fistula_rehab.db

# Run the training pipeline by default
CMD ["python", "-m", "src.training.trainer"]