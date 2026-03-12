import sqlite3
import faiss
import numpy as np
import logging
import json
import os
from typing import Tuple, Optional, List
from datetime import datetime

logger = logging.getLogger("IdentityStore")

class IdentityStore:
    def __init__(self, db_path: str = "memory/agent.db", faiss_index_path: str = "memory/faces.index", embedding_dim: int = 128):
        self.db_path = db_path
        self.faiss_path = faiss_index_path
        self.dim = embedding_dim
        
        # Ensure dir exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Init SQLite
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_db()
        
        # Init FAISS
        if os.path.exists(faiss_index_path):
            try:
                self.index = faiss.read_index(faiss_index_path)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors.")
                if self.index.d != self.dim:
                    logger.warning(f"Existing index dim {self.index.d} != {self.dim}. Creating new.")
                    self.index = faiss.IndexFlatIP(self.dim)
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}. Creating new.")
                self.index = faiss.IndexFlatIP(self.dim)
        else:
            self.index = faiss.IndexFlatIP(self.dim)

    def _init_db(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS identities (
                person_id TEXT PRIMARY KEY,
                first_seen TEXT,
                last_seen TEXT,
                consent INTEGER DEFAULT 0,
                emotion_history TEXT
            )
        ''')
        self.conn.commit()

    def find_or_create(self, embedding: np.ndarray, threshold: float = 0.55) -> Tuple[str, bool]:
        """
        Returns (person_id, is_new).
        embedding: (128,) float32 normalized vector
        """
        # FAISS search
        if self.index.ntotal > 0:
            # Reshape for faiss
            query = embedding.reshape(1, -1).astype('float32')
            D, I = self.index.search(query, 1)
            
            sim = D[0][0]
            idx = I[0][0]
            
            if sim > threshold:
                person_id = f"person_{idx}"
                self._update_seen(person_id)
                return person_id, False

        # Create new
        new_idx = self.index.ntotal
        self.index.add(embedding.reshape(1, -1).astype('float32'))
        faiss.write_index(self.index, self.faiss_path)
        
        person_id = f"person_{new_idx}"
        now = datetime.now().isoformat()
        
        self.cursor.execute(
            "INSERT INTO identities (person_id, first_seen, last_seen, consent, emotion_history) VALUES (?, ?, ?, ?, ?)",
            (person_id, now, now, 0, json.dumps([]))
        )
        self.conn.commit()
        
        logger.info(f"Created new identity: {person_id}")
        return person_id, True

    def _update_seen(self, person_id: str):
        now = datetime.now().isoformat()
        self.cursor.execute("UPDATE identities SET last_seen = ? WHERE person_id = ?", (now, person_id))
        self.conn.commit()

    def get_consent(self, person_id: str) -> bool:
        res = self.cursor.execute("SELECT consent FROM identities WHERE person_id = ?", (person_id,)).fetchone()
        return bool(res[0]) if res else False

    def set_consent(self, person_id: str, allowed: bool):
        val = 1 if allowed else 0
        self.cursor.execute("UPDATE identities SET consent = ? WHERE person_id = ?", (val, person_id))
        self.conn.commit()

    def close(self):
        self.conn.close()
