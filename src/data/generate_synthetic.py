import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uuid
import random
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm
from src.data.noise_injection import inject_noise
from src.data.database import DatabaseManager
from src.utils.logger import get_logger
from src.utils.helpers import load_config

logger = get_logger(__name__)

# Load config
try:
    config = load_config()
except Exception as e:
    logger.error(f"Failed to load config: {e}")
    sys.exit(1)

SYMPTOM_TEMPLATES = {
    0: [
        "no leakage today, feeling fine",
        "slight wetness but no pain",
        "catheter ok, drinking water",
        "sawa tu, hakuna shida",
        "small water but not worried",
        "dry all day, very happy"
    ],
    1: [
        "leaking small amount each hour",
        "pain when sitting, no fever",
        "wound looks red but no smell",
        "naumwa kidogo tumbo",
        "catheter blocked, can't pee",
        "little bit of water, but no smell"
    ],
    2: [
        "fever 38.5 and wound smells bad",
        "bleeding a lot, dizzy",
        "can't urinate at all since yesterday",
        "hari mai na zazzabi",
        "pain so bad I cannot walk",
        "green discharge from wound, very sick"
    ]
}

RARE_DANGEROUS = [
    ("wound smells but no pain", 2),
    ("no urine output for 2 days, no pain", 2),
    ("confused and weak after surgery", 2)
]

LANGUAGES = ['en', 'sw', 'ha']
CODE_SWITCH_PHRASES = ['tafadhali saidia', 'allah ya taimake', 'please help', 'asante']

def generate_patient(patient_id: str) -> dict:
    return {
        "patient_id": patient_id,
        "age_group": np.random.choice(['15-24','25-34','35-44','45+'], p=[0.3,0.4,0.2,0.1]),
        "fistula_type": np.random.choice(['VVF','RVF','mixed'], p=[0.7,0.2,0.1]),
        "repair_date": (datetime.now() - timedelta(days=np.random.randint(1,90))).isoformat(),
        "region": np.random.choice(['pilot_north','pilot_south','pilot_east'])
    }

def generate_symptom_report(patient_id: str, timestamp: datetime) -> dict:
    # Choose class using numpy but convert to plain Python int
    cls_np = np.random.choice([0,1,2], p=[0.60,0.25,0.15])
    cls = int(cls_np)   # ensure Python int

    if cls == 2 and np.random.random() < config['data_generation']['rare_urgent_oversample']:
        text, cls = random.choice(RARE_DANGEROUS)   # cls is already Python int (2)
    else:
        text = random.choice(SYMPTOM_TEMPLATES[cls])

    text = inject_noise(text, prob=config['data_generation']['noise_probability'])
    if np.random.random() < config['data_generation']['code_switch_probability']:
        text += " " + random.choice(CODE_SWITCH_PHRASES)

    language = np.random.choice(LANGUAGES)

    return {
        "patient_id": patient_id,
        "raw_message": text,
        "language": language,
        "timestamp": timestamp.isoformat(),
        "predicted_class": None,
        "prediction_confidence": None,
        "human_review_class": cls,   # guaranteed Python int
        "human_review_timestamp": timestamp.isoformat(),
        "action_taken": None
    }

def generate_and_store(db: DatabaseManager):
    n_patients = config['data_generation']['n_synthetic_patients']
    n_reports = config['data_generation']['n_symptom_reports']
    logger.info(f"Generating {n_patients} synthetic patients and {n_reports} reports")
    patient_ids = []

    # Insert patients
    for _ in tqdm(range(n_patients), desc="Patients"):
        pid = str(uuid.uuid4())
        patient_ids.append(pid)
        db.insert_patient(generate_patient(pid))

    # Insert reports
    start_date = datetime.now() - timedelta(days=60)
    for _ in tqdm(range(n_reports), desc="Reports"):
        patient_id = random.choice(patient_ids)
        rand_days = random.randint(0, 60)
        rand_hours = random.randint(0, 23)
        rand_minutes = random.randint(0, 59)
        timestamp = start_date + timedelta(days=rand_days, hours=rand_hours, minutes=rand_minutes)
        report = generate_symptom_report(patient_id, timestamp)
        db.insert_symptom_report(report)

    logger.info("Synthetic data generation complete")

if __name__ == "__main__":
    # Delete old db for a clean start
    import os
    db_path = config['database']['path']
    if os.path.exists(db_path):
        os.remove(db_path)

    db = DatabaseManager(db_path)
    generate_and_store(db)

    # Verify insertion with integer check
    with db.get_connection() as conn:
        count_patients = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        count_reports = conn.execute("SELECT COUNT(*) FROM symptom_reports").fetchone()[0]
        # Check that all human_review_class values are integers
        sample = conn.execute("SELECT human_review_class FROM symptom_reports LIMIT 5").fetchall()
        logger.info(f"Sample labels: {[row[0] for row in sample]}")
        logger.info(f"Verification: {count_patients} patients, {count_reports} symptom reports in database.")

    if count_reports == 0:
        logger.error("No reports inserted! Check previous errors.")
        sys.exit(1)