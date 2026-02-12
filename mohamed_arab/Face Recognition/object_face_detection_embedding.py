import cv2
import torch
import time
import json
import os
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import torch.nn.functional as F

# =========================
# تحميل النماذج
# =========================
yolo_face_model = YOLO("yolov8n-face.pt")
yolo_object_model = YOLO("yolov8n.pt")
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# =========================
# تجهيز الفولدرات
# =========================
os.makedirs("faces", exist_ok=True)
os.makedirs("objects", exist_ok=True)

# =========================
# تحميل قاعدة البيانات
# =========================
def load_db(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return {}

def save_db(path, data):
    with open(path, "w") as f:
        json.dump(data, f)

facenet_db = load_db("facenet_db.json")

# =========================
# FaceNet Embedding
# =========================
def embed_facenet(img):
    img = cv2.resize(img, (160,160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.tensor(img).permute(2,0,1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        return facenet(t)[0]

def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# =========================
# تسجيل وجه جديد
# =========================
def register_new_user(face_crop):
    name = input("Enter person's name: ").strip()
    emb = embed_facenet(face_crop).tolist()
    facenet_db[name] = emb
    save_db("facenet_db.json", facenet_db)

    save_path = os.path.join("faces", f"{name}.jpg")
    cv2.imwrite(save_path, face_crop)

    print(f"\nFace saved as: {name}\n")

# =========================
# IoU Function
# =========================
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    bx1, by1, bx2, by2 = box2

    inter_area = max(0, min(x2, bx2) - max(x1, bx1)) * \
                 max(0, min(y2, by2) - max(y1, by1))

    box1_area = (x2-x1)*(y2-y1)

    if box1_area == 0:
        return 0

    return inter_area / box1_area


# =========================
# تشغيل الكاميرا
# =========================
cap = cv2.VideoCapture(0)

print("Press R to register a new face - Press Q to quit")

saved_objects = []
object_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =========================
    # Face Detection
    # =========================
    face_results = yolo_face_model(frame, verbose=False)[0]

    for box in face_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        emb = embed_facenet(face_crop)

        best_name = None
        best_sim = -1

        for name, saved_emb in facenet_db.items():
            sim = cosine_sim(emb, torch.tensor(saved_emb))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        label = best_name if best_sim > 0.55 else "Unknown"

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # =========================
    # Object Detection
    # =========================
    object_results = yolo_object_model(frame, verbose=False)[0]

    for box, cls_id in zip(object_results.boxes, object_results.boxes.cls):

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = object_results.names[int(cls_id)]
        object_crop = frame[y1:y2, x1:x2]

        duplicate = False

        for saved_box in saved_objects:
            if compute_iou((x1,y1,x2,y2), saved_box) > 0.5:
                duplicate = True
                break

        # رسم البوكس دايماً
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(frame, class_name, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        # الحفظ مرة واحدة فقط
        if not duplicate and object_crop.size > 0:
            object_counter += 1
            save_path = os.path.join("objects", f"{class_name}_{object_counter}.jpg")
            cv2.imwrite(save_path, object_crop)
            saved_objects.append((x1,y1,x2,y2))

    cv2.imshow("Face + Object Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        print("Capturing face for registration...")
        time.sleep(0.5)
        ret, frame = cap.read()
        results_reg = yolo_face_model(frame, verbose=False)[0]
        if len(results_reg.boxes) == 0:
            print("No face detected! Try again.")
            continue
        x1, y1, x2, y2 = map(int, results_reg.boxes[0].xyxy[0])
        face_crop = frame[y1:y2, x1:x2]
        register_new_user(face_crop)

cap.release()
cv2.destroyAllWindows()