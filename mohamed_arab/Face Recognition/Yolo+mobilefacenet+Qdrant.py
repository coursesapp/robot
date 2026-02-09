import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import torch.nn.functional as F
import uuid

# ------------------------------
# Models
# ------------------------------
yolo_model = YOLO("yolov8n-face.pt")
mobilefacenet = torch.jit.load(
    "C:\\Users\\Mohamed\\.cache\\torch\\checkpoints\\mobilefacenet_scripted.pt"
).eval()

# ------------------------------
# Qdrant setup
# ------------------------------
COLLECTION_NAME = "faces_db"
client = QdrantClient(url="http://localhost:6333")

# تحقق من وجود collection، لو مش موجود اعمله
existing = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME not in existing:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=128,
            distance=Distance.COSINE
        )
    )

# ------------------------------
# Functions
# ------------------------------
def embed_mobilefacenet(img):
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        emb = mobilefacenet(t)[0]
    return F.normalize(emb, dim=0).cpu().numpy()

def register_new_user(face_crop):
    name = input("Enter person's name: ").strip()
    embedding = embed_mobilefacenet(face_crop)
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding.tolist(),
        payload={"name": name}
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[point])
    print(f"\nFace saved in Qdrant as: {name}\n")

def recognize_face(embedding, threshold=0.4):
    try:
        # الطريقة الصحيحة في Qdrant 1.16
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding.tolist(),
            limit=1
        )
        if not results:
            return None, 0.0
        best = results[0]
        if best.score < threshold:
            return None, 0.0
        return best.payload["name"], best.score
    except Exception as e:
        print("Error during search:", e)
        return None, 0.0

def show_all_embeddings():
    print("\n========== EMBEDDINGS IN QDRANT ==========")
    points = client.scroll(collection_name=COLLECTION_NAME, limit=100)[0]
    if not points:
        print("No embeddings stored.")
        return
    for i, p in enumerate(points, 1):
        vec_preview = p.vector[:5] if p.vector else []
        print(f"{i}. Name: {p.payload['name']}")
        print(f"   Vector (first 5): {vec_preview} ...\n")

def delete_all_embeddings():
    confirm = input("Are you sure you want to delete ALL embeddings? (y/n): ").strip().lower()
    if confirm == "y":
        client.delete_collection(collection_name=COLLECTION_NAME)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=128, distance=Distance.COSINE)
        )
        print("All embeddings deleted.\n")

# ------------------------------
# Camera loop
# ------------------------------
cap = cv2.VideoCapture(0)
print("R: Register | E: Show Embeddings | D: Delete All | Q: Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        t1 = time.time()
        emb = embed_mobilefacenet(face_crop)
        name, score = recognize_face(emb)
        t2 = time.time()

        label = "Unknown"
        if name:
            label = f"{name} ({score:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        print("\n====================")
        print("Face Detected")
        print(f"Match: {label}")
        print(f"Inference Time: {(t2 - t1)*1000:.2f} ms")

    cv2.imshow("Live Face Recognition - Qdrant", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        time.sleep(0.5)
        ret, frame = cap.read()
        results_reg = yolo_model(frame, verbose=False)[0]
        if len(results_reg.boxes) == 0:
            print("No face detected!")
            continue
        x1, y1, x2, y2 = map(int, results_reg.boxes[0].xyxy[0])
        face_crop = frame[y1:y2, x1:x2]
        register_new_user(face_crop)
    elif key == ord('e'):
        show_all_embeddings()
    elif key == ord('d'):
        delete_all_embeddings()

cap.release()
cv2.destroyAllWindows()