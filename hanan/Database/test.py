import numpy as np
from db.memory_manager import MemoryManager

def manual_embedding_test():

    config_path = "config/config.json"

    memory_manager = MemoryManager(config_path, config_path)

    # USER 1 
    user1 = {
        "name": "Karim",
        "preferences": ["software engineering"],
        "habits": "working late",
        "schedule": "8 pm - 4 am"
    }

    face_emb1 = np.random.rand(512).tolist()
    voice_emb1 = np.random.rand(512).tolist()
    text_emb1 = np.random.rand(512).tolist()

    print("\nStoring USER 1...\n")
    user1_id = memory_manager.store_user_embedding(
        user1,
        face_emb1,
        voice_emb1,
        text_emb1
    )

    print("\nUser1 MongoDB ID:", user1_id)


    # USER 2
    user2 = {
        "name": "mona",
        "preferences": ["AI", "robotics"],
        "habits": "studying late",
        "schedule": "7 pm - 3 am"
    }

    face_emb2 = np.random.rand(512).tolist()
    voice_emb2 = np.random.rand(512).tolist()
    text_emb2 = np.random.rand(512).tolist()

    print("\nStoring USER 2...\n")
    user2_id = memory_manager.store_user_embedding(
        user2,
        face_emb2,
        voice_emb2,
        text_emb2
    )

    print("\nUser2 MongoDB ID:", user2_id)


    # SEARCH TEST 
    print("\nSearching using USER 2 text embedding...")

    query_emb = text_emb1   # exact match with user1

    search_result = memory_manager.qdrant_manager.search_embedding(
        "text_embeddings",
        query_emb,
        limit=1
    )

    print("\nRaw Qdrant Result:", search_result)


    if search_result.get("result"):
        top_match = search_result["result"][0]

        ref_id = top_match["payload"]["reference_id"]

        print("\nMatched reference_id:", ref_id)

        retrieved_user = memory_manager.mongo_manager.get_user_data(ref_id)

        print("\nRetrieved User from MongoDB:", retrieved_user)

    else:
        print("No results found")


if __name__ == "__main__":
    manual_embedding_test()