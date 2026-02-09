import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO

# 1) Load YOLOv8 face detection
print("Loading YOLOv8n-face...")
yolo_model = YOLO("yolov8n-face.pt")

# 2) Load FaceNet
print("Loading FaceNet...")
facenet = InceptionResnetV1(pretrained='vggface2').eval()

last_embedding = None
EMBEDDING_THRESHOLD = 0.9

def get_embedding(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_tensor = torch.tensor(face_img).float().permute(2,0,1) / 255.0
    face_tensor = face_tensor.unsqueeze(0)
    with torch.no_grad():
        emb = facenet(face_tensor)
    return emb

def is_same_embedding(emb1, emb2):
    if emb1 is None or emb2 is None:
        return False
    distance = torch.nn.functional.pairwise_distance(emb1, emb2).item()
    return distance < EMBEDDING_THRESHOLD

# Open Camera
cap = cv2.VideoCapture(0)
print("Camera started... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection (disable default verbose printing)
    results = yolo_model(frame, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        embedding = get_embedding(face_crop)

        if not is_same_embedding(last_embedding, embedding):
            print("\nFace = embedding:")
            print(embedding)  # هنا بيطبع الـ embedding كامل
            last_embedding = embedding

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.imshow("Live Face Detection + FaceNet", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
