import sqlite3
import pandas as pd
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import struct
import json

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
            
            # === Module A: Triage Tables ===
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

            # === Module B: Chat Tables ===
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    patient_id TEXT,
                    state_json TEXT,
                    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_activity TEXT DEFAULT CURRENT_TIMESTAMP,
                    phq9_score INTEGER,
                    phq9_severity TEXT,
                    FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    direction TEXT CHECK(direction IN ('user', 'bot')),
                    message TEXT NOT NULL,
                    intent TEXT,
                    is_crisis BOOLEAN DEFAULT 0,
                    confidence REAL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id)
                )
            ''')

            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_messages_timestamp ON chat_messages(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_symptom_reports_patient ON symptom_reports(patient_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_symptom_reports_timestamp ON symptom_reports(timestamp)')

    # ============================================================
    # Module A: Patient Management
    # ============================================================
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

    def get_patient(self, patient_id: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            row = conn.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,)).fetchone()
            return dict(row) if row else None

    # ============================================================
    # Module A: Symptom Reports
    # ============================================================
    def insert_symptom_report(self, report: Dict[str, Any]) -> int:
        if 'human_review_class' in report and report['human_review_class'] is not None:
            report['human_review_class'] = int(report['human_review_class'])
        try:
            with self.get_connection() as conn:
                cursor = conn.execute('''
                    INSERT INTO symptom_reports 
                    (patient_id, raw_message, language, timestamp, predicted_class, prediction_confidence, 
                     human_review_class, human_review_timestamp, action_taken)
                    VALUES (:patient_id, :raw_message, :language, :timestamp, :predicted_class, 
                            :prediction_confidence, :human_review_class, :human_review_timestamp, :action_taken)
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
        if val is None:
            return None
        if isinstance(val, bytes):
            if len(val) == 8:
                return struct.unpack('<q', val)[0]
            else:
                return int.from_bytes(val, byteorder='little')
        return int(val)

    def get_training_data(self, min_confidence: float = 0.0) -> pd.DataFrame:
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
        df['label'] = df['label'].apply(self._convert_label)
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        return df

    def log_inference(self, report_id: int, model_version: str, predicted_class: int, 
                      confidence: float, latency_ms: float) -> None:
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO inference_logs (report_id, model_version, inference_timestamp, 
                                            predicted_class, confidence, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (report_id, model_version, datetime.now().isoformat(), 
                  predicted_class, confidence, latency_ms))

    # ============================================================
    # Module B: Chat Sessions
    # ============================================================
    def create_chat_session(self, session_id: str, patient_id: Optional[str] = None) -> None:
        with self.get_connection() as conn:
            conn.execute('''
                INSERT OR IGNORE INTO chat_sessions (session_id, patient_id)
                VALUES (?, ?)
            ''', (session_id, patient_id))

    def save_chat_state(self, session_id: str, state: dict) -> None:
        with self.get_connection() as conn:
            conn.execute('''
                UPDATE chat_sessions
                SET state_json = ?, last_activity = CURRENT_TIMESTAMP
                WHERE session_id = ?
            ''', (json.dumps(state), session_id))

    def load_chat_state(self, session_id: str) -> Optional[dict]:
        with self.get_connection() as conn:
            row = conn.execute('''
                SELECT state_json FROM chat_sessions WHERE session_id = ?
            ''', (session_id,)).fetchone()
        if row and row[0]:
            return json.loads(row[0])
        return None

    def log_chat_message(self, session_id: str, direction: str, message: str, 
                         intent: Optional[str] = None, is_crisis: bool = False,
                         confidence: Optional[float] = None) -> None:
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO chat_messages (session_id, direction, message, intent, is_crisis, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, direction, message, intent, is_crisis, confidence))

    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        with self.get_connection() as conn:
            rows = conn.execute('''
                SELECT direction, message, intent, is_crisis, timestamp
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (session_id, limit)).fetchall()
            return [dict(row) for row in reversed(rows)]

    def update_phq9_score(self, session_id: str, score: int, severity: str) -> None:
        with self.get_connection() as conn:
            conn.execute('''
                UPDATE chat_sessions
                SET phq9_score = ?, phq9_severity = ?
                WHERE session_id = ?
            ''', (score, severity, session_id))

    def get_active_sessions(self, age_minutes: int = 60) -> List[str]:
        """Return session IDs that have been active in the last N minutes."""
        with self.get_connection() as conn:
            rows = conn.execute('''
                SELECT session_id FROM chat_sessions
                WHERE datetime(last_activity) > datetime('now', ?)
            ''', (f'-{age_minutes} minutes',)).fetchall()
            return [row[0] for row in rows]

    def get_chat_session_stats(self) -> Dict[str, Any]:
        """Get statistics about chat sessions and messages."""
        stats = {}
        with self.get_connection() as conn:
            # Total sessions
            row = conn.execute('SELECT COUNT(*) FROM chat_sessions').fetchone()
            stats['total_sessions'] = row[0] if row else 0
            
            # Total messages
            row = conn.execute('SELECT COUNT(*) FROM chat_messages').fetchone()
            stats['total_messages'] = row[0] if row else 0
            
            # Crisis messages
            row = conn.execute('SELECT COUNT(*) FROM chat_messages WHERE is_crisis = 1').fetchone()
            stats['crisis_messages'] = row[0] if row else 0
            
            # Intent distribution
            rows = conn.execute('''
                SELECT intent, COUNT(*) as count FROM chat_messages 
                WHERE intent IS NOT NULL AND direction = 'user'
                GROUP BY intent
            ''').fetchall()
            stats['intent_distribution'] = {row[0]: row[1] for row in rows}
            
            # PHQ-9 scores
            rows = conn.execute('''
                SELECT phq9_severity, COUNT(*) FROM chat_sessions 
                WHERE phq9_severity IS NOT NULL
                GROUP BY phq9_severity
            ''').fetchall()
            stats['phq9_distribution'] = {row[0]: row[1] for row in rows}
            
        return stats

    # ============================================================
    # Utility Methods
    # ============================================================
    def execute_raw_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute a raw query and return results as list of dicts."""
        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def vacuum(self) -> None:
        """Optimize the database."""
        with self.get_connection() as conn:
            conn.execute('VACUUM')