import hashlib
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)

class EmbeddingCache:
    def __init__(self, db_path: str, model_name: str):
        self.db_path = db_path
        self.model_name = model_name
        self._init_cache_table()
    
    def _init_cache_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    message_hash TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    model_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    def _hash_message(self, message: str) -> str:
        return hashlib.sha256(message.encode()).hexdigest()
    
    def get(self, message: str) -> np.ndarray | None:
        h = self._hash_message(message)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute('SELECT embedding FROM embedding_cache WHERE message_hash = ? AND model_name = ?', 
                               (h, self.model_name)).fetchone()
            if row:
                return np.frombuffer(row[0], dtype=np.float32)
        return None
    
    def set(self, message: str, embedding: np.ndarray):
        h = self._hash_message(message)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('INSERT OR REPLACE INTO embedding_cache (message_hash, embedding, model_name) VALUES (?, ?, ?)',
                         (h, embedding.tobytes(), self.model_name))

class TextEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_db_path: str = "data/fistula_rehab.db"):
        self.model = SentenceTransformer(model_name)
        self.cache = EmbeddingCache(cache_db_path, model_name)
        self.model_name = model_name
    
    def embed(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        embeddings = []
        uncached_indices = []
        uncached_texts = []
        for i, text in enumerate(texts):
            if use_cache:
                emb = self.cache.get(text)
                if emb is not None:
                    embeddings.append(emb)
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Compute missing embeddings in batch
        if uncached_texts:
            new_embs = self.model.encode(uncached_texts, show_progress_bar=False)
            for idx, emb in zip(uncached_indices, new_embs):
                embeddings.insert(idx, emb)
                if use_cache:
                    self.cache.set(uncached_texts[uncached_indices.index(idx)], emb)
        return np.array(embeddings)