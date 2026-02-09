import cv2
import torch
import time
import json
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import torch.nn.functional as F

yolo_model = YOLO("yolov8n-face.pt")
facenet = InceptionResnetV1(pretrained='vggface2').eval()
mobilefacenet = torch.jit.load("C:\\Users\\Mohamed\\.cache\\torch\\checkpoints\\mobilefacenet_scripted.pt").eval()

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
mobilefacenet_db = load_db("mobilefacenet_db.json")

def embed_facenet(img):
    img = cv2.resize(img, (160,160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.tensor(img).permute(2,0,1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        return facenet(t)[0]

def embed_mobilefacenet(img):
    img = cv2.resize(img, (112,112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.tensor(img).permute(2,0,1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        return mobilefacenet(t)[0]

def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def register_new_user(face_crop):
    name = input("Enter person's name: ").strip()
    emb_fn = embed_facenet(face_crop).tolist()
    emb_mfn = embed_mobilefacenet(face_crop).tolist()
    facenet_db[name] = emb_fn
    mobilefacenet_db[name] = emb_mfn
    save_db("facenet_db.json", facenet_db)
    save_db("mobilefacenet_db.json", mobilefacenet_db)
    print(f"\nFace saved as: {name}\n")

cap = cv2.VideoCapture(0)
print("Press R to register a new face - Press Q to quit")

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
        facenet_emb = embed_facenet(face_crop)
        t2 = time.time()
        facenet_time = (t2 - t1) * 1000

        t3 = time.time()
        mfn_emb = embed_mobilefacenet(face_crop)
        t4 = time.time()
        mobilefacenet_time = (t4 - t3) * 1000

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        best_name_fn, best_sim_fn = None, -1
        best_name_mfn, best_sim_mfn = None, -1

        for name, saved_emb in facenet_db.items():
            sim = cosine_sim(facenet_emb, torch.tensor(saved_emb))
            if sim > best_sim_fn:
                best_sim_fn = sim
                best_name_fn = name

        for name, saved_emb in mobilefacenet_db.items():
            sim = cosine_sim(mfn_emb, torch.tensor(saved_emb))
            if sim > best_sim_mfn:
                best_sim_mfn = sim
                best_name_mfn = name

        print("\n====================")
        print("Face Detected")
        print("--------------------")
        print(f"[FaceNet] Best Match: {best_name_fn} | Similarity: {best_sim_fn:.4f} | Time: {facenet_time:.2f} ms")
        print(f"[MobileFaceNet] Best Match: {best_name_mfn} | Similarity: {best_sim_mfn:.4f} | Time: {mobilefacenet_time:.2f} ms")

    cv2.imshow("Live Face Recognizer", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        print("Capturing face for registration...")
        time.sleep(0.5)
        ret, frame = cap.read()
        results_reg = yolo_model(frame, verbose=False)[0]

        if len(results_reg.boxes) == 0:
            print("No face detected! Try again.")
            continue

        x1, y1, x2, y2 = map(int, results_reg.boxes[0].xyxy[0])
        face_crop = frame[y1:y2, x1:x2]
        register_new_user(face_crop)

cap.release()
cv2.destroyAllWindows()