import cv2
import torch
import time
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import torch.nn.functional as F

# ===== YOLOv8 face detection =====
yolo_model = YOLO("yolov8n-face.pt")

# ===== FaceNet =====
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# ===== MobileFaceNet =====
mobilefacenet_path = "C:\\Users\\Mohamed\\.cache\\torch\\checkpoints\\mobilefacenet_scripted.pt"
mobilefacenet = torch.load(mobilefacenet_path, map_location='cpu')
mobilefacenet.eval()

last_embedding_facenet = None
last_embedding_mfn = None
EMBEDDING_THRESHOLD = 0.9  # مسافة للتمييز بين الـ embeddings المختلفة

def get_embedding_facenet(face_img):
    face_img = cv2.resize(face_img, (160,160))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_tensor = torch.tensor(face_img).permute(2,0,1).unsqueeze(0).float() / 255.0
    start_time = time.time()
    with torch.no_grad():
        emb = facenet(face_tensor)
    elapsed = time.time() - start_time
    return emb, elapsed

def get_embedding_mfn(face_img):
    face_img = cv2.resize(face_img, (112,112))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_tensor = torch.tensor(face_img).permute(2,0,1).unsqueeze(0).float() / 255.0
    start_time = time.time()
    with torch.no_grad():
        emb = mobilefacenet(face_tensor)
    elapsed = time.time() - start_time
    return emb, elapsed

def compare_embeddings(emb1, emb2):
    """احسب التشابه بين الـ embeddings (cosine similarity)"""
    similarity = F.cosine_similarity(emb1, emb2).item()
    return similarity

# ===== Open Camera =====
cap = cv2.VideoCapture(0)

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

        # ===== Get embeddings & times =====
        emb_facenet, time_fn = get_embedding_facenet(face_crop)
        emb_mfn, time_mfn = get_embedding_mfn(face_crop)

        # ===== Compare with last embeddings to avoid تكرار الطباعة =====
        print("\n===== New Face Detected =====")
        print(f"[FaceNet] embedding: {emb_facenet}\nTime: {time_fn:.4f}s")
        print(f"[MobileFaceNet] embedding: {emb_mfn}\nTime: {time_mfn:.4f}s")

        similarity = compare_embeddings(emb_facenet, emb_mfn)
        print(f"Cosine similarity between FaceNet & MobileFaceNet: {similarity:.4f}")

        # ===== Draw bounding box =====
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.imshow("YOLO + FaceNet + MobileFaceNet", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
