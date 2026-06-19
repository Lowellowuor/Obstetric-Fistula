import joblib
import time
import numpy as np
from src.features.embedder import TextEmbedder
from src.data.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TriagePredictor:
    def __init__(self, model_path: str, db_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = joblib.load(model_path)
        self.embedder = TextEmbedder(model_name=embedding_model, cache_db_path=db_path)
        self.db = DatabaseManager(db_path)
    
    def predict(self, patient_id: str, message: str, language: str = "en") -> dict:
        start_time = time.time()
        report_data = {
            "patient_id": patient_id,
            "raw_message": message,
            "language": language,
            "timestamp": None,
            "predicted_class": None,
            "prediction_confidence": None,
            "human_review_class": None,
            "human_review_timestamp": None,
            "action_taken": None
        }
        report_id = self.db.insert_symptom_report(report_data)
        
        emb = self.embedder.embed([message])[0].reshape(1, -1)
        proba = self.model.predict_proba(emb)[0]
        predicted_class = int(np.argmax(proba))
        confidence = float(np.max(proba))
        
        self.db.update_prediction(report_id, predicted_class, confidence)
        
        if predicted_class == 2:
            action = "URGENT: Notify clinician immediately"
        elif predicted_class == 1:
            action = "Watchful: Monitor and report if worsens"
        else:
            action = "Routine: Continue recovery"
        
        latency_ms = (time.time() - start_time) * 1000
        self.db.log_inference(report_id, "xgboost_v1", predicted_class, confidence, latency_ms)
        
        return {
            "report_id": report_id,
            "triage_class": predicted_class,
            "confidence": confidence,
            "action_recommended": action
        }
