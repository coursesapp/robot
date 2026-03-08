import requests
import json
import uuid


class QdrantManager:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)

        self.qdrant_uri = config["qdrant_uri"]
        self.collection_names = config["qdrant_collection_names"]

        # Create collections if they don't exist
        for col_name in self.collection_names.values():
            self.create_collection_if_not_exists(col_name)

    def create_collection_if_not_exists(self, collection_name, vector_size=512, distance="Cosine"):
        response = requests.get(f"{self.qdrant_uri}/collections/{collection_name}")

        if response.status_code == 404:
            payload = {
                "vectors": {
                    "size": vector_size,
                    "distance": distance
                }
            }

            resp = requests.put(
                f"{self.qdrant_uri}/collections/{collection_name}",
                json=payload  
            )

            print(f"Collection {collection_name} created:", resp.json())
        else:
            print(f"Collection {collection_name} already exists.")

    # Convert Mongo _id → Deterministic UUID
    def mongo_id_to_uuid(self, mongo_id):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(mongo_id)))

    def store_embedding(self, collection_name, embedding, reference_id):

        # Convert embedding to list if numpy array
        if not isinstance(embedding, list):
            embedding = embedding.tolist()

        # Convert Mongo ID to UUID
        qdrant_id = self.mongo_id_to_uuid(reference_id)

        payload = {
            "points": [
                {
                    "id": qdrant_id,  
                    "vector": embedding,
                    "payload": {
                        "reference_id": str(reference_id)
                    }
                }
            ]
        }

        response = requests.put(
            f"{self.qdrant_uri}/collections/{collection_name}/points",
            json=payload 
        )

        print("Store embedding response:", response.json())
        return response.json()

    def search_embedding(self, collection_name, query_embedding, limit=5):

        if not isinstance(query_embedding, list):
            query_embedding = query_embedding.tolist()

        payload = {
            "vector": query_embedding,
            "limit": limit,
            "with_payload": True   
        }

        response = requests.post(
            f"{self.qdrant_uri}/collections/{collection_name}/points/search",
            json=payload 
        )

        return response.json()