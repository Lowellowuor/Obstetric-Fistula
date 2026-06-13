import numpy as np
import joblib
from src.features.embedder import TextEmbedder
from src.data.database import DatabaseManager

class UncertaintySampler:
    def __init__(self, db_path: str, model_path: str):
        self.db = DatabaseManager(db_path)
        self.model = joblib.load(model_path)
        self.embedder = TextEmbedder(cache_db_path=db_path)
    
    def query_uncertain(self, batch_size: int = 20) -> list:
        reports = self.db.get_unlabeled_reports_for_active_learning(limit=100)
        if not reports:
            return []
        texts = [r['raw_message'] for r in reports]
        X = self.embedder.embed(texts)
        proba = self.model.predict_proba(X)
        # Margin uncertainty
        margins = 1 - (np.max(proba, axis=1) - np.partition(proba, -2, axis=1)[:,-2])
        uncertain_indices = np.argsort(-margins)[:batch_size]
        selected = [reports[i] for i in uncertain_indices if margins[i] > 0.3]
        for rep in selected:
            self.db.update_human_review(rep['report_id'], None)  # mark for review
        return selected