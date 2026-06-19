import joblib
import numpy as np
from src.features.embedder import TextEmbedder
from src.utils.helpers import load_config

class CrisisDetector:
    def __init__(self, model_path: str, db_path: str, config_path: str = "config/phase2_config.yaml"):
        self.model = joblib.load(model_path)
        self.embedder = TextEmbedder(cache_db_path=db_path)
        self.config = load_config(config_path)
        self.keywords = self.config['crisis_detection']['keywords']
        self.confidence_threshold = self.config['crisis_detection']['confidence_threshold']
        self.intent_map = {v: k for k, v in enumerate(self.config['chat']['intents'])}
        self.crisis_id = self.intent_map["crisis"]
    
    def detect(self, message: str) -> tuple[bool, float]:
        """Return (is_crisis, confidence_score)."""
        # Step 1: Keyword check
        message_lower = message.lower()
        for kw in self.keywords:
            if kw in message_lower:
                return True, 1.0  # immediate flag
        
        # Step 2: ML confidence check (only if no keyword)
        emb = self.embedder.embed([message])[0].reshape(1, -1)
        proba = self.model.predict_proba(emb)[0]
        crisis_prob = float(proba[self.crisis_id])
        
        if crisis_prob >= self.confidence_threshold:
            return True, crisis_prob
        
        return False, crisis_prob