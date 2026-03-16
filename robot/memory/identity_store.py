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
    def __init__(self, 
                 db_path: str = "memory/agent.db", 
                 faiss_index_path: str = "memory/faces.index", 
                 embedding_dim: int = 128,
                 voice_index_path: str = "memory/voices.index",
                 voice_dim: int = 256):
        self.db_path = db_path
        self.faiss_path = faiss_index_path
        self.dim = embedding_dim
        self.voice_faiss_path = voice_index_path
        self.voice_dim = voice_dim
        
        # Ensure dir exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Init SQLite
        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL") # Enable WAL mode for concurrency
        self._init_db()
        
        # Init FAISS
        if os.path.exists(faiss_index_path):
            try:
                self.index = faiss.read_index(faiss_index_path)
                logger.info(f"Loaded Face FAISS index with {self.index.ntotal} vectors.")
                if self.index.d != self.dim:
                    logger.warning(f"Face index dim {self.index.d} != {self.dim}. Creating new.")
                    self.index = faiss.IndexFlatIP(self.dim)
            except Exception as e:
                logger.error(f"Failed to load Face FAISS index: {e}. Creating new.")
                self.index = faiss.IndexFlatIP(self.dim)
        else:
            self.index = faiss.IndexFlatIP(self.dim)

        # Init Voice FAISS
        if os.path.exists(voice_index_path):
            try:
                self.voice_index = faiss.read_index(voice_index_path)
                logger.info(f"Loaded Voice FAISS index with {self.voice_index.ntotal} vectors.")
            except Exception as e:
                logger.error(f"Failed to load Voice FAISS index: {e}")
                self.voice_index = faiss.IndexFlatIP(self.voice_dim)
        else:
            self.voice_index = faiss.IndexFlatIP(self.voice_dim)

    def _init_db(self):
        cursor = self.conn.cursor()
        # Table for permanent identities
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS identities (
                person_id TEXT PRIMARY KEY,
                first_seen TEXT,
                last_seen TEXT,
                consent INTEGER DEFAULT 0,
                emotion_history TEXT
            )
        ''')
        # Table mapping FAISS indices to identities (Face)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_templates (
                faiss_idx INTEGER PRIMARY KEY,
                person_id TEXT,
                view_type TEXT, -- 'frontal', 'side'
                FOREIGN KEY(person_id) REFERENCES identities(person_id)
            )
        ''')
        # Table mapping FAISS indices to identities (Voice)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS voice_templates (
                faiss_idx INTEGER PRIMARY KEY,
                person_id TEXT,
                FOREIGN KEY(person_id) REFERENCES identities(person_id)
            )
        ''')
        self.conn.commit()
        cursor.close()

    def find_or_create(self, 
                       embedding: np.ndarray, 
                       threshold: float = 0.35, 
                       create: bool = True,
                       person_id: Optional[str] = None,
                       is_frontal: bool = True) -> Tuple[str, bool, float]:
        """
        Returns (person_id, is_new, confidence).
        If person_id is provided, it ADDS this embedding to that specific person.
        is_frontal: Used to decide if we should allow creating a BRAND NEW person.
        """
        # 1. Search existing templates
        if self.index.ntotal > 0:
            query = embedding.reshape(1, -1).astype('float32')
            D, I = self.index.search(query, 1)
            sim = D[0][0]
            template_idx = int(I[0][0])
            
            if sim > threshold:
                # Map template_idx back to real person_id
                cursor = self.conn.cursor()
                res = cursor.execute("SELECT person_id FROM face_templates WHERE faiss_idx = ?", (template_idx,)).fetchone()
                cursor.close()
                if res:
                    matched_id = res[0]
                    # logger.info(f"MATCH: {matched_id} (score {sim:.4f})")
                    self._update_seen(matched_id)
                    return matched_id, False, float(sim)
            
            if sim > 0.25: # Still within gray zone
                return "unknown", False, float(sim)

        # 2. If track-locked ID is provided, just add this as a new template for them
        if person_id and person_id != "unknown":
            logger.info(f"LEARNING: Adding new {('frontal' if is_frontal else 'side')} view for {person_id}")
            self._add_template(person_id, embedding, is_frontal)
            return person_id, False, 1.0 # Assigned confidence is max

        # 3. Create NEW person (Only allowed if frontal and create=True)
        if create and is_frontal:
            new_person_id = f"person_{self._get_next_person_idx()}"
            now = datetime.now().isoformat()
            
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO identities (person_id, first_seen, last_seen, consent, emotion_history) VALUES (?, ?, ?, ?, ?)",
                (new_person_id, now, now, 0, json.dumps([]))
            )
            self._add_template(new_person_id, embedding, is_frontal)
            self.conn.commit()
            cursor.close()
            
            logger.info(f"NEW IDENTITY: Created {new_person_id} from frontal view.")
            return new_person_id, True, 1.0

        return "unknown", False, 0.0

    def find_or_create_voice(self, embedding: np.ndarray, threshold: float = 0.7, person_id: Optional[str] = None) -> Tuple[str, bool, float]:
        """Matches a voice embedding and returns (person_id, is_new, confidence)."""
        # 1. Search existing templates
        matched_id = "unknown"
        sim = 0.0
        if self.voice_index.ntotal > 0:
            query = embedding.reshape(1, -1).astype('float32')
            D, I = self.voice_index.search(query, 1)
            sim = D[0][0]
            template_idx = int(I[0][0])
            
            if sim > threshold:
                cursor = self.conn.cursor()
                res = cursor.execute("SELECT person_id FROM voice_templates WHERE faiss_idx = ?", (template_idx,)).fetchone()
                cursor.close()
                if res:
                    matched_id = res[0]
                    
        # 2. Logic: If vision ALREADY knows who this is, and voice suggests someone else,
        # we prioritize Vision if it's a stable identification, or use Voice only for unknown.
        if person_id and person_id != "unknown":
            # If voice matched a DIFFERENT person, we might have a conflict.
            # For now, if vision is confident, we link this voice to the vision ID as a new template
            # (unless voice match was extremely high > 0.85)
            if matched_id != "unknown" and matched_id != person_id:
                logger.debug(f"Fusion Conflict: Voice thinks {matched_id}, Vision thinks {person_id}. Trusting Vision.")
                # We won't return matched_id, we'll proceed to link voice to person_id
            else:
                if matched_id == person_id:
                    self._update_seen(matched_id)
                    return matched_id, False, float(sim)
                
                # Logic: Link this voice to the person we see ONLY if they have < 5 templates
                cursor = self.conn.cursor()
                count = cursor.execute("SELECT COUNT(*) FROM voice_templates WHERE person_id = ?", (person_id,)).fetchone()[0]
                cursor.close()
                
                if count < 5:
                    logger.info(f"LEARNING: Linking new voice template to {person_id} (count={count})")
                    id, is_new = self._link_voice(person_id, embedding)
                    return id, is_new, 1.0
                else:
                    logger.debug(f"Voice Learning: Max templates reached for {person_id}. Skipping.")
                    return person_id, False, 0.5 # Low confidence fallback

        if matched_id != "unknown":
            self._update_seen(matched_id)
            return matched_id, False, float(sim)

        return "unknown", False, float(sim)

    def _link_voice(self, person_id: str, embedding: np.ndarray) -> Tuple[str, bool]:
        # Adaptive dimension check
        if self.voice_index.ntotal == 0 and embedding.shape[0] != self.voice_dim:
            logger.warning(f"Voice embedding dim {embedding.shape[0]} != {self.voice_dim}. Re-creating index.")
            self.voice_dim = embedding.shape[0]
            self.voice_index = faiss.IndexFlatIP(self.voice_dim)
        
        try:
            faiss_idx = self.voice_index.ntotal
            self.voice_index.add(embedding.reshape(1, -1).astype('float32'))
            faiss.write_index(self.voice_index, self.voice_faiss_path)
            cursor = self.conn.cursor()
            cursor.execute("INSERT INTO voice_templates (faiss_idx, person_id) VALUES (?, ?)", (faiss_idx, person_id))
            self.conn.commit()
            cursor.close()
            return person_id, True # is_new=True
        except Exception as e:
            logger.error(f"Failed to add voice template: {e}")
            return "unknown", False

        return "unknown", False

    def _add_template(self, person_id: str, embedding: np.ndarray, is_frontal: bool):
        faiss_idx = self.index.ntotal
        self.index.add(embedding.reshape(1, -1).astype('float32'))
        faiss.write_index(self.index, self.faiss_path)
        
        view_type = 'frontal' if is_frontal else 'side'
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO face_templates (faiss_idx, person_id, view_type) VALUES (?, ?, ?)",
            (faiss_idx, person_id, view_type)
        )
        self.conn.commit()
        cursor.close()

    def _get_next_person_idx(self) -> int:
        cursor = self.conn.cursor()
        res = cursor.execute("SELECT COUNT(*) FROM identities").fetchone()
        cursor.close()
        return res[0] if res else 0

    def _update_seen(self, person_id: str):
        now = datetime.now().isoformat()
        cursor = self.conn.cursor()
        cursor.execute("UPDATE identities SET last_seen = ? WHERE person_id = ?", (now, person_id))
        self.conn.commit()
        cursor.close()

    def get_consent(self, person_id: str) -> bool:
        cursor = self.conn.cursor()
        res = cursor.execute("SELECT consent FROM identities WHERE person_id = ?", (person_id,)).fetchone()
        cursor.close()
        return bool(res[0]) if res else False

    def set_consent(self, person_id: str, allowed: bool):
        val = 1 if allowed else 0
        cursor = self.conn.cursor()
        cursor.execute("UPDATE identities SET consent = ? WHERE person_id = ?", (val, person_id))
        self.conn.commit()
        cursor.close()

    def close(self):
        self.conn.close()
