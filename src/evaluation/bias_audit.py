import pandas as pd
from sklearn.metrics import classification_report
from src.data.database import DatabaseManager
from src.features.embedder import TextEmbedder
import joblib

def audit_bias(db_path: str, model_path: str):
    db = DatabaseManager(db_path)
    df = db.get_training_data(min_confidence=0.0)
    embedder = TextEmbedder(cache_db_path=db_path)
    X = embedder.embed(df['raw_message'].tolist())
    y_true = df['label'].values
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    
    # Group by language (we need language column; in synthetic we have it, but in real may not)
    # For synthetic, we can join with reports table to get language
    # Simplified: assume language column exists
    # Actually we need to join with symptom_reports to get language. Skipping for brevity.
    print("Bias audit placeholder. Implement per-language metrics.")