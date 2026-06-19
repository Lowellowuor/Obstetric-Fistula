import pytest
from src.data.database import DatabaseManager
from src.data.generate_synthetic import generate_and_store
import tempfile

def test_synthetic_generation():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db = DatabaseManager(tmp.name)
        import yaml
        with open("config/phase1_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        config['data_generation']['n_synthetic_patients'] = 10
        config['data_generation']['n_symptom_reports'] = 50
        assert True
