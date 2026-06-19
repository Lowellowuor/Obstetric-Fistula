import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import mlflow
import joblib
from src.features.embedder import TextEmbedder
from src.data.database import DatabaseManager
from src.evaluation.clinical_metrics import clinical_safety_metrics
from src.utils.logger import get_logger
from src.utils.helpers import load_config

logger = get_logger(__name__)

def train_xgboost():
    config = load_config()
    db_path = config['database']['path']
    db = DatabaseManager(db_path)
    
    df = db.get_training_data(min_confidence=0.0)
    logger.info(f"Training data size: {len(df)}")
    
    if len(df) == 0:
        logger.error("No training data found. Please run data generation first.")
        return None
    
    df['label'] = df['label'].astype(int)
    
    embedder = TextEmbedder(
        model_name=config['features']['text_embedding_model'], 
        cache_db_path=db_path
    )
    X = embedder.embed(df['raw_message'].tolist())
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    scale_pos_weight = {0: 1.0, 1: 2.0, 2: config['model']['xgboost_params']['scale_pos_weight_urgent']}
    sample_weights = np.array([scale_pos_weight[label] for label in y_train])
    
    model = xgb.XGBClassifier(
        objective=config['model']['xgboost_params']['objective'],
        num_class=config['model']['xgboost_params']['num_class'],
        max_depth=config['model']['xgboost_params']['max_depth'],
        learning_rate=config['model']['xgboost_params']['learning_rate'],
        n_estimators=config['model']['xgboost_params']['n_estimators'],
        eval_metric='mlogloss',
        random_state=42
    )
    
    with mlflow.start_run(run_name="xgboost_training"):
        mlflow.log_params(config['model']['xgboost_params'])
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            verbose=True
        )
        y_pred = model.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            target_names=['routine', 'watchful', 'urgent'],
            output_dict=True
        )
        safety = clinical_safety_metrics(y_test, y_pred)
        mlflow.log_metrics({
            "accuracy": report['accuracy'],
            "urgent_recall": safety['urgent_sensitivity'],
            "urgent_precision": safety['urgent_ppv'],
            "false_negative_rate": safety['false_negative_rate']
        })
        joblib.dump(model, "models_artifacts/xgboost_model.pkl")
        mlflow.sklearn.log_model(model, "xgboost_model")
        logger.info(f"Model saved. Urgent recall: {safety['urgent_sensitivity']:.3f}")
    
    return model

if __name__ == "__main__":
    train_xgboost()
