import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from src.models.predict import TriagePredictor
from src.data.schemas import PredictionRequest, PredictionResponse
import yaml

app = FastAPI(title="Fistula Triage API")

# Load config
with open("config/phase1_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize predictor (lazy loading)
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = TriagePredictor(
            model_path="models_artifacts/xgboost_model.pkl",
            db_path=config['database']['path'],
            embedding_model=config['features']['text_embedding_model']
        )
    return predictor

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    pred = get_predictor()
    result = pred.predict(request.patient_id, request.message, request.language)
    return PredictionResponse(
        report_id=result['report_id'],
        triage_class=result['triage_class'],
        confidence=result['confidence'],
        action_recommended=result['action_recommended'],
        explanation=None
    )

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)