from db.mongo_manager import MongoManager
from db.qdrant_manager import QdrantManager

class MemoryManager:
    def __init__(self, mongo_config_path, qdrant_config_path):
        self.mongo_manager = MongoManager(mongo_config_path)
        self.qdrant_manager = QdrantManager(qdrant_config_path)

    def store_user_embedding(self, user_data, face_embedding, voice_embedding, text_embedding):
        # Store user info in MongoDB
        user_id = self.mongo_manager.store_user_data(user_data)

        # Store embeddings in Qdrant
        self.qdrant_manager.store_embedding("identity_embeddings", face_embedding, user_id)
        self.qdrant_manager.store_embedding("identity_embeddings", voice_embedding, user_id)
        self.qdrant_manager.store_embedding("text_embeddings", text_embedding, user_id)

        return user_id

    def retrieve_user_info(self, query_embedding):
        # Search for the most similar embeddings in Qdrant
        search_result = self.qdrant_manager.search_embedding("text_embeddings", query_embedding)

        # Ensure the result contains the expected 'reference_id'
        if "result" in search_result and len(search_result["result"]) > 0:
            reference_id = search_result["result"][0]["payload"]["reference_id"]
            # Use the `reference_id` to fetch the corresponding user info from MongoDB
            user_info = self.mongo_manager.get_user_data(reference_id)
            return user_info
        else:
            print("No results found in Qdrant search.")
            return None