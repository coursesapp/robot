import logging
import uuid
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from typing import List, Dict, Any

logger = logging.getLogger("VectorMemory")

class VectorMemory:
    def __init__(self, db_path: str, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Load embedding model
        logger.info(f"Loading Vector Embedding Model ({model_name})...")
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # We use a single main collection for the agent, filtering by person_id
        self.collection_name = "agent_long_term_memory"
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        logger.info(f"Vector Database connected. Total interactions cached: {self.collection.count()}")

    def add_interaction(self, person_id: str, text: str, role: str = "user"):
        """Save a single line or long text to the vector DB. Long texts are chunk-split."""
        if not text or not text.strip():
            return

        timestamp = time.time()
        time_str = time.strftime("%Y-%m-%d %H:%M:%S")

        # Split only if text is long (summaries, documents), keep short dialogue as-is
        if len(text) > 500:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
            chunks = splitter.split_text(text)
        else:
            chunks = [text]

        for chunk in chunks:
            doc_id = str(uuid.uuid4())
            metadata = {
                "person_id": person_id,
                "role": role,
                "timestamp": timestamp,
                "time_str": time_str
            }
            vector = self.embeddings_model.embed_query(chunk)
            self.collection.add(
                embeddings=[vector],
                documents=[chunk],
                metadatas=[metadata],
                ids=[doc_id]
            )
        logger.debug(f"Saved {len(chunks)} chunk(s) to VectorMemory for {person_id}: {text[:30]}...")

    def search_past(self, person_id: str, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Search past interactions specifically for this person."""
        if self.collection.count() == 0:
            return []
            
        query_vector = self.embeddings_model.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            where={"person_id": person_id}
        )
        
        formatted_results = []
        if results and results['documents'] and results['documents'][0]:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            
            for doc, meta in zip(docs, metas):
                formatted_results.append({
                    "text": doc,
                    "role": meta.get("role", "unknown"),
                    "time_str": meta.get("time_str", "unknown")
                })
                
        # Sort by timestamp chronologically if we want
        # formatted_results.sort(key=lambda x: x['timestamp'])
        
        return formatted_results
