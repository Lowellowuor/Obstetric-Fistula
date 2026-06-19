# AI-Enabled Post-Surgical Triage & Psychosocial Support – MVP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

##  Overview

**A Minimum Viable Product (MVP) for AI-assisted post-operative monitoring and psychosocial support for women recovering from obstetric fistula surgery in Sub-Saharan Africa.**

This platform provides:

1. **Intelligent post-surgical triage** (`/predict`) – classifies patient-reported symptoms into **Routine**, **Watchful**, or **Urgent** to enable faster clinical intervention.
2. **Psychosocial support chatbot** (`/chat`) – delivers empathetic conversations, detects crisis (suicidal ideation), and administers a culturally adapted **PHQ-9 depression screening**.

Both modules share a unified **FastAPI** server, a single **SQLite** database, and a common embedding cache. The system is designed for low-bandwidth, offline-first environments and is ready for SMS/WhatsApp integration.

---

##  What We Have Achieved

| Module   | Feature                                                                         | Status        |
| -------- | ------------------------------------------------------------------------------- | ------------- |
| **A**    | Symptom triage (Routine/Watchful/Urgent)                                        | ✅ Complete    |
| **A**    | Cost-sensitive XGBoost model (urgent recall **99.3%**)                          | ✅ Trained     |
| **A**    | Synthetic data generation (5,000 reports, 200 patients)                         | ✅ Complete    |
| **A**    | Active learning loop (uncertainty sampling for clinician review)                | ✅ Implemented |
| **B**    | Intent classification (6 intents: coping, stigma, info, peer, crisis, greeting) | ✅ Trained     |
| **B**    | Crisis detection (keyword + ML confidence, zero false negatives)                | ✅ Operational |
| **B**    | Dialogue state machine (opening → follow-up → closing)                          | ✅ Operational |
| **B**    | PHQ-9 depression screening (9 questions, severity scoring)                      | ✅ Operational |
| **B**    | Curated response retrieval (empathic, safe, culturally adapted)                 | ✅ Deployed    |
| **Both** | Unified FastAPI server (`/predict` + `/chat`)                                   | ✅ Running     |
| **Both** | SQLite persistence (patients, triage logs, chat sessions, PHQ-9 scores)         | ✅ Operational |
| **Both** | MLflow experiment tracking                                                      | ✅ Integrated  |
| **Both** | Docker containerisation                                                         | ✅ Supported   |

---

##  Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/your-org/fistula-rehab-platform.git
cd fistula-rehab-platform

python -m venv venv

# Linux / Mac
source venv/bin/activate

# Windows PowerShell
venv\Scripts\Activate.ps1

pip install -r requirements-dev.txt
```

### 2. Generate Synthetic Data

#### Module A – Triage Data

```bash
python -m src.data.generate_synthetic
```

Creates:

```text
data/fistula_rehab.db
```

with 200 patients and 5,000 symptom reports.

#### Module B – Chat Data

```bash
python -m src.data.generate_chat_data
```

Generates:

```text
data/synthetic/chat_data.csv
```

with 10,000 English messages across 6 intents.

### 3. Train Models

#### Module A – Triage Model

```bash
python -m src.models.train_xgboost
```

Saves:

```text
models_artifacts/xgboost_model.pkl
```

#### Module B – Intent Classifier

```bash
python -m src.models.train_intent
```

Saves:

```text
models_artifacts/intent_model.pkl
```

### 4. Run the Unified API Server

#### Windows PowerShell

```powershell
$env:PYTHONPATH="."
python scripts/start_api.py
```

#### Linux / Mac

```bash
export PYTHONPATH=.
python scripts/start_api.py
```

API available at:

```text
http://localhost:8000
```

Interactive Swagger docs:

```text
http://localhost:8000/docs
```

### 5. Test a Prediction (Module A – Triage)

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"patient_id":"test123","message":"fever and wound smells bad","language":"en"}'
```

Response:

```json
{
  "report_id": 5001,
  "triage_class": 2,
  "confidence": 0.99,
  "action_recommended": "URGENT: Notify clinician immediately",
  "explanation": null
}
```

### 6. Test a Chat (Module B – Psychosocial Support)

```powershell
# Test 1 – Emotional support
$body = '{"session_id":"s1","message":"I feel very sad and lonely"}'
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/chat" -ContentType "application/json" -Body $body

# Test 2 – Crisis detection
$body = '{"session_id":"s2","message":"I want to kill myself"}'
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/chat" -ContentType "application/json" -Body $body

# Test 3 – PHQ-9 depression screening
$body = '{"session_id":"s3","message":"depression test"}'
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/chat" -ContentType "application/json" -Body $body
```

---

##  Performance Summary

### Module A – Triage (Synthetic Test Set, n=1,000)

| Class        | Precision | Recall | F1-Score |
| ------------ | --------- | ------ | -------- |
| Routine (0)  | 0.98      | 0.99   | 0.99     |
| Watchful (1) | 0.96      | 0.95   | 0.95     |
| Urgent (2)   | 0.99      | 0.99   | 0.99     |

**Urgent Recall (Clinical Sensitivity):** 0.993

**False Negative Rate for Urgent:** 0.007

### Module B – Chat (Intent Classification)

| Metric               | Value |
| -------------------- | ----- |
| Accuracy             | ~0.89 |
| Crisis Recall        | 1.00  |
| Confidence Threshold | 0.4   |

>  Note: All results are based on synthetic data. Real-world performance will be validated during the pilot phase.

---

##  Running Tests

### Unit & Safety Tests

```bash
pytest tests/ -v
```

### Manual End-to-End Test

```powershell
.\test_module_b.ps1
```

---

##  Docker Deployment

```bash
docker build -t fistula-platform .
docker run -p 8000:8000 fistula-platform
```

---

##  MLflow Dashboard

```bash
mlflow ui --backend-store-uri ./models_artifacts/mlflow
```

Open:

```text
http://localhost:5000
```

---

##  Project Structure

```text
fistula-rehab-platform/
├── config/
│   ├── phase1_config.yaml
│   └── phase2_config.yaml
├── data/
│   ├── fistula_rehab.db
│   ├── responses_en.json
│   └── synthetic/
├── models_artifacts/
│   ├── xgboost_model.pkl
│   ├── intent_model.pkl
│   └── mlflow/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── chat/
│   │   ├── dialogue_manager.py
│   │   ├── crisis_detector.py
│   │   └── phq9.py
│   ├── training/
│   ├── evaluation/
│   ├── active_learning/
│   └── utils/
├── scripts/
│   └── start_api.py
├── tests/
├── Dockerfile
├── Makefile
└── requirements.txt
```

---

##  Limitations and Known Issues

| Limitation                         | Workaround / Future Plan                           |
| ---------------------------------- | -------------------------------------------------- |
| Trained only on synthetic data     | Collect real patient messages and fine-tune models |
| Chat responses are in English only | Expand to Swahili, Hausa, Amharic                  |
| No SMS/WhatsApp gateway            | Integrate Twilio / Africa's Talking                |
| Requires internet for inference    | Convert models to ONNX / TensorFlow Lite           |
| Not clinically approved            | Complete ethics and regulatory approvals           |

---

##  Next Steps (Modules C & D)

| Module | Description                                                                     | Status    |
| ------ | ------------------------------------------------------------------------------- | --------- |
| C      | Economic Reintegration – personalised skills matching, micro-enterprise toolkit |  Planned |
| D      | Policy & Research Dashboard – analytics and systematic literature review        |  Planned |
| Pilot  | Real-world deployment with 20–40 patients and clinicians                        |  Planned |

---

##  Contributing

This is an open-source humanitarian project.

Contributions are welcome through:

* Pull Requests
* Feature Requests
* Bug Reports
* Documentation Improvements

---

##  License

MIT License – free for humanitarian, educational, research, and non-commercial use.

---

##  Acknowledgements

Built with  for women's health in Sub-Saharan Africa.
