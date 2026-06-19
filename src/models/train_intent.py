import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import mlflow
import joblib
from src.features.embedder import TextEmbedder
from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_intent_classifier():
    config = load_config("config/phase2_config.yaml")
    df = pd.read_csv("data/synthetic/chat_data.csv")
    logger.info(f"Training data size: {len(df)}")
    
    embedder = TextEmbedder(
        model_name="all-MiniLM-L6-v2",
        cache_db_path="data/fistula_rehab.db"
    )
    X = embedder.embed(df['text'].tolist())
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    scale_pos_weight = config['intent_model']['xgboost_params']['scale_pos_weight']
    sample_weights = np.array([scale_pos_weight[label] for label in y_train])
    
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=6,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        eval_metric="mlogloss",
        random_state=42
    )
    
    with mlflow.start_run(run_name="intent_classifier"):
        mlflow.log_params(config['intent_model']['xgboost_params'])
        model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=config['chat']['intents'], output_dict=True)
        mlflow.log_metrics({"accuracy": report['accuracy']})
        joblib.dump(model, "models_artifacts/intent_model.pkl")
        mlflow.sklearn.log_model(model, "intent_model")
        logger.info(f"Intent model saved. Accuracy: {report['accuracy']:.3f}")
    
    return model

if __name__ == "__main__":
    train_intent_classifier()