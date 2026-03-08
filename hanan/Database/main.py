from db.memory_manager import MemoryManager
from modules.face_recognition import FaceRecognition
from modules.voice_recognition import VoiceRecognition
from modules.text_embedding import TextEmbedding

def main():
    # Configuration paths
    mongo_config_path = "config/config.json"
    qdrant_config_path = "config/config.json"

    # Initialize memory manager and modules
    memory_manager = MemoryManager(mongo_config_path, qdrant_config_path)
    face_recognition = FaceRecognition()
    voice_recognition = VoiceRecognition()
    text_embedding = TextEmbedding()

    # Example user data
    user_data = {
        "name": "John Doe",
        "preferences": ["robotics", "AI"],
        "habits": "exercises daily",
        "schedule": "9 AM - 5 PM"
    }

    # these should come from actual modules
    face_image = None  # Replace with actual face image
    audio_clip = None  # Replace with actual audio clip
    text = "Hello, I am John."

    # Generate embeddings
    face_emb = face_recognition.get_embedding(face_image)
    voice_emb = voice_recognition.get_embedding(audio_clip)
    text_emb = text_embedding.get_embedding(text)

    # Store the data
    user_id = memory_manager.store_user_embedding(user_data, face_emb, voice_emb, text_emb)
    print(f"Stored user with ID: {user_id}")

    # Retrieve user information using a text query 
    query_text = "Hello, I need information about John."
    query_emb = text_embedding.get_embedding(query_text)
    user_info = memory_manager.retrieve_user_info(query_emb)
    print(f"Retrieved user info: {user_info}")

if __name__ == "__main__":
    main()