from src.data.database import DatabaseManager
from src.models.train_xgboost import train_xgboost
from src.active_learning.uncertainty_sampling import UncertaintySampler
from src.utils.logger import get_logger
import yaml

logger = get_logger(__name__)

def run_training_pipeline():
    with open("config/phase1_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    logger.info("Starting training pipeline")
    model = train_xgboost()
    
    # Active learning after training
    sampler = UncertaintySampler(config['database']['path'], "models_artifacts/xgboost_model.pkl")
    uncertain = sampler.query_uncertain(batch_size=config['active_learning']['batch_size'])
    logger.info(f"Queued {len(uncertain)} reports for clinician review")
    
    return model

if __name__ == "__main__":
    run_training_pipeline()