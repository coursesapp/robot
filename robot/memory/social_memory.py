import sqlite3
import logging
import json
from typing import Dict, Any, Optional

logger = logging.getLogger("SocialMemory")

class SocialMemory:
    def __init__(self, db_path: str = "memory/agent.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL") # Enable WAL mode for concurrency
        self._init_db()

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS social (
                person_id TEXT PRIMARY KEY,
                name TEXT,
                interests TEXT,   -- JSON List
                job TEXT,
                summary TEXT      -- JSON List of summaries
            )
        ''')
        self.conn.commit()
        cursor.close()

    def get(self, person_id: str) -> Dict[str, Any]:
        cursor = self.conn.cursor()
        row = cursor.execute(
            "SELECT name, interests, job, summary FROM social WHERE person_id = ?", 
            (person_id,)
        ).fetchone()
        cursor.close()
        
        if row:
            return {
                "name": row[0],
                "interests": json.loads(row[1]) if row[1] else [],
                "job": row[2],
                "summary": json.loads(row[3]) if row[3] else []
            }
        return {}

    def update(self, person_id: str, data: Dict[str, Any]):
        current = self.get(person_id)
        
        # Merge fields and ensure they are strings
        name = data.get('name', current.get('name'))
        if isinstance(name, list): name = ", ".join(name)
        
        job = data.get('job', current.get('job'))
        if isinstance(job, list): job = ", ".join(job)
        
        interests = current.get('interests', [])
        if 'interests' in data:
            interests.extend(data['interests'])
            interests = list(set(interests)) # Unique
            
        summary = current.get('summary', [])
        if 'summary' in data:
            summary.append(data['summary']) 
            
        # Upsert
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO social (person_id, name, interests, job, summary)
            VALUES (?, ?, ?, ?, ?)
        ''', (person_id, name, json.dumps(interests), job, json.dumps(summary)))
        self.conn.commit()
        cursor.close()
