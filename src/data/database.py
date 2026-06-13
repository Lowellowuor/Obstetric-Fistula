import sqlite3
import pandas as pd
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import struct

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_tables()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_tables(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    age_group TEXT,
                    fistula_type TEXT,
                    repair_date TEXT,
                    region TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS symptom_reports (
                    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    raw_message TEXT NOT NULL,
                    language TEXT,
                    timestamp TEXT,
                    predicted_class INTEGER,
                    prediction_confidence REAL,
                    human_review_class INTEGER,
                    human_review_timestamp TEXT,
                    action_taken TEXT,
                    FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    message_hash TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    model_name TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS inference_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id INTEGER,
                    model_version TEXT,
                    inference_timestamp TEXT,
                    predicted_class INTEGER,
                    confidence REAL,
                    latency_ms REAL,
                    FOREIGN KEY(report_id) REFERENCES symptom_reports(report_id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drift_metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT,
                    metric_value REAL,
                    window_start TEXT,
                    window_end TEXT,
                    recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def insert_patient(self, patient: Dict[str, Any]) -> None:
        try:
            with self.get_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO patients (patient_id, age_group, fistula_type, repair_date, region)
                    VALUES (:patient_id, :age_group, :fistula_type, :repair_date, :region)
                ''', patient)
        except Exception as e:
            logger.error(f"Failed to insert patient: {e}")
            logger.error(f"Patient data: {patient}")
            raise

    def insert_symptom_report(self, report: Dict[str, Any]) -> int:
        # Ensure human_review_class is Python int
        if 'human_review_class' in report and report['human_review_class'] is not None:
            report['human_review_class'] = int(report['human_review_class'])
        try:
            with self.get_connection() as conn:
                cursor = conn.execute('''
                    INSERT INTO symptom_reports 
                    (patient_id, raw_message, language, timestamp, predicted_class, prediction_confidence, human_review_class, human_review_timestamp, action_taken)
                    VALUES (:patient_id, :raw_message, :language, :timestamp, :predicted_class, :prediction_confidence, :human_review_class, :human_review_timestamp, :action_taken)
                ''', report)
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to insert report: {e}")
            logger.error(f"Report data: {report}")
            raise

    def update_prediction(self, report_id: int, predicted_class: int, confidence: float) -> None:
        with self.get_connection() as conn:
            conn.execute('''
                UPDATE symptom_reports
                SET predicted_class = ?, prediction_confidence = ?
                WHERE report_id = ?
            ''', (predicted_class, confidence, report_id))

    def get_unlabeled_reports_for_active_learning(self, limit: int = 100) -> List[Dict]:
        with self.get_connection() as conn:
            rows = conn.execute('''
                SELECT report_id, raw_message, predicted_class, prediction_confidence
                FROM symptom_reports
                WHERE human_review_class IS NULL AND predicted_class IS NOT NULL
                ORDER BY prediction_confidence ASC
                LIMIT ?
            ''', (limit,)).fetchall()
            return [dict(row) for row in rows]

    def update_human_review(self, report_id: int, corrected_label: Optional[int]) -> None:
        with self.get_connection() as conn:
            conn.execute('''
                UPDATE symptom_reports
                SET human_review_class = ?, human_review_timestamp = CURRENT_TIMESTAMP
                WHERE report_id = ?
            ''', (corrected_label, report_id))

    def _convert_label(self, val):
        """Convert label from possible bytes or other types to int."""
        if val is None:
            return None
        if isinstance(val, bytes):
            # Try to decode as little-endian 64-bit integer (common for SQLite)
            if len(val) == 8:
                return struct.unpack('<q', val)[0]
            else:
                # Fallback: try to convert via int from string
                return int.from_bytes(val, byteorder='little')
        return int(val)

    def get_training_data(self, min_confidence: float = 0.0) -> pd.DataFrame:
        """Retrieve data with labels: either human-reviewed or high-confidence predictions."""
        with self.get_connection() as conn:
            query = '''
                SELECT raw_message, 
                       COALESCE(human_review_class, predicted_class) as label,
                       prediction_confidence
                FROM symptom_reports
                WHERE human_review_class IS NOT NULL 
                   OR (predicted_class IS NOT NULL AND prediction_confidence >= ?)
            '''
            df = pd.read_sql_query(query, conn, params=(min_confidence,))
        
        # Convert label column using the robust converter
        df['label'] = df['label'].apply(self._convert_label)
        # Drop rows where label became None (shouldn't happen)
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        return df

    def log_inference(self, report_id: int, model_version: str, predicted_class: int, confidence: float, latency_ms: float) -> None:
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO inference_logs (report_id, model_version, inference_timestamp, predicted_class, confidence, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (report_id, model_version, datetime.now().isoformat(), predicted_class, confidence, latency_ms))