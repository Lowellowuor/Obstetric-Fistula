# AI-Enabled Post-Surgical Triage for Obstetric Fistula – Phase One

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

##  Overview

**A Minimum Viable Product (MVP) for AI-assisted post-operative monitoring of women recovering from obstetric fistula surgery in Sub-Saharan Africa.**

This platform provides intelligent post-surgical triage using machine learning to classify patient-reported symptoms into urgency categories, enabling faster intervention and better recovery outcomes in low-resource settings.

The system is designed for:

* Low-bandwidth environments
* Offline-first deployment
* REST API integration
* Future SMS / WhatsApp connectivity

---

#  What We Have Achieved (Phase One)

We have built, trained, and deployed a **production-ready AI triage system** that classifies patient-reported symptoms into three urgency levels:

* **Routine** – continue standard recovery
* **Watchful** – monitor at home and report if symptoms worsen
* **Urgent** – immediate clinical intervention required

---

##  Key Accomplishments

| Feature                                                                 | Status        |
| ----------------------------------------------------------------------- | ------------- |
| Synthetic data generation (5,000 reports, 200 patients)                 | ✅ Complete    |
| Realistic noise injection (typos, code-switching, rare dangerous cases) | ✅ Complete    |
| Cost-sensitive XGBoost model (urgent recall **99.3%**)                  | ✅ Trained     |
| Embedding cache (avoids recomputation)                                  | ✅ Implemented |
| SQLite database with active learning queue                              | ✅ Operational |
| FastAPI inference endpoint                                              | ✅ Running     |
| MLflow experiment tracking                                              | ✅ Integrated  |
| Unit & safety tests                                                     | ✅ Passing     |
| Docker containerisation                                                 | ✅ Supported   |
| Comprehensive documentation                                             | ✅ Provided    |

---

#  Quick Start

## 1. Clone & Setup

```bash
git clone https://github.com/your-org/fistula-phase1.git
cd fistula-phase1

python -m venv venv

# Linux / Mac
source venv/bin/activate

# Windows PowerShell
venv\Scripts\Activate.ps1

pip install -r requirements-dev.txt
```

---

## 2. Generate Synthetic Data

```bash
python -m src.data.generate_synthetic
```

Creates:

```text
data/fistula_rehab.db
```

with:

* 200 patients
* 5,000 symptom reports

---

## 3. Train the Model

```bash
python -m src.models.train_xgboost
```

Artifacts generated:

```text
models_artifacts/xgboost_model.pkl
```

Metrics are automatically logged to MLflow.

---

## 4. Run the API Server

### Windows PowerShell

```powershell
$env:PYTHONPATH="."
python scripts/start_api.py
```

### Linux / Mac

```bash
export PYTHONPATH=.
python scripts/start_api.py
```

API available at:

```text
http://localhost:8000
```

Interactive Swagger documentation:

```text
http://localhost:8000/docs
```

---

## 5. Test a Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"test123","message":"fever and wound smells bad","language":"en"}'
```

### Example Response

```json
{
  "report_id": 5001,
  "triage_class": 2,
  "confidence": 0.93,
  "action_recommended": "URGENT: Notify clinician immediately",
  "explanation": null
}
```

---

#  Performance Summary

Evaluation on a held-out synthetic test set (`n = 1,000`).

| Class        | Precision | Recall | F1-score |
| ------------ | --------- | ------ | -------- |
| Routine (0)  | 0.98      | 0.99   | 0.99     |
| Watchful (1) | 0.96      | 0.95   | 0.95     |
| Urgent (2)   | 0.99      | 0.99   | 0.99     |

### Clinical Metrics

* **Urgent Recall (Clinical Sensitivity):** `0.993`
* **False Negative Rate for Urgent Cases:** `0.007`

>  These results are based on synthetic data. Real-world performance may vary.

---

#  Running Tests

Run all tests:

```bash
pytest tests/ -v
```

Run a quick manual validation:

```bash
python dirty_test.py
```

---

#  Docker Deployment

## Build Image

```bash
docker build -t fistula-triage .
```

## Run Container

```bash
docker run -p 8000:8000 fistula-triage
```

---

#  MLflow Dashboard

View experiment tracking, metrics, parameters, and artifacts.

```bash
mlflow ui --backend-store-uri ./models_artifacts/mlflow
```

Open in browser:

```text
http://localhost:5000
```

---

#  Project Structure

```text
fistula-phase1/
├── config/                 # YAML configuration
├── data/                   # SQLite DB, synthetic CSV
├── models_artifacts/       # Trained model, MLflow logs
├── src/
│   ├── data/               # Database, schemas, data generation
│   ├── features/           # Text embedding with caching
│   ├── models/             # XGBoost training & prediction
│   ├── training/           # Unified pipeline
│   ├── evaluation/         # Clinical metrics, bias audit
│   ├── active_learning/    # Uncertainty sampling
│   └── utils/              # Logging, config helper
├── tests/                  # Unit & safety tests
├── scripts/                # API server, pipeline runner
├── Dockerfile
├── Makefile
└── requirements.txt
```

---

#  Limitations and Known Issues

| Limitation                            | Workaround / Future Plan                                     |
| ------------------------------------- | ------------------------------------------------------------ |
| Trained only on synthetic data        | Collect and label real patient messages from partner clinics |
| No SMS gateway integration            | Integrate Twilio or Africa’s Talking API                     |
| Supports English, Swahili, Hausa only | Expand multilingual support using multilingual BERT          |
| Requires internet for inference       | Convert model to ONNX / TensorFlow Lite                      |
| Not clinically approved               | Submit ethics and regulatory approvals before pilot          |

---

#  Next Steps (Phase Two)

## Planned Enhancements

###  Systematic Literature Review

* PRISMA-guided review of AI in fistula care

###  Real-World Pilot

* Collect 200+ real patient messages
* Fine-tune production models

###  Psychosocial Support Module

* Intent classification
* PHQ-9 mental health screening

###  SMS Integration

* Two-way patient messaging

###  User Acceptance Study

* Interviews with nurses and survivors

---

#  Contributing

This is an open-source humanitarian project.

Contributions are welcome through:

* Pull requests
* Feature requests
* Bug reports
* Documentation improvements

Please open an issue before major changes.

---

#  License

This project is licensed under the **MIT License**.

Free for:

* Humanitarian use
* Educational use
* Research use
* Non-commercial deployment

See the `LICENSE` file for details.

---
