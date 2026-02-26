import cv2
import torch
import time
import json
import os
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch.nn.functional as F
from collections import deque
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Models (VISION — بدون تعديل)
# =========================
yolo_face_model = YOLO("yolov8n-face.pt")
yolo_object_model = YOLO("yolov8n.pt")

mtcnn = MTCNN(image_size=160, margin=20, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# =========================
# BLIP LOCAL MODEL
# =========================
print("Loading BLIP locally...")
model_path = "./blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_path)
blip_model = BlipForConditionalGeneration.from_pretrained(model_path).to(device)
print("BLIP loaded successfully!")

# =========================
# Database
# =========================
def load_db(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return {k: F.normalize(torch.tensor(v).to(device), p=2, dim=0)
                    for k, v in data.items()}
    except:
        return {}

def save_db(path, data):
    serializable = {k: v.cpu().tolist() for k, v in data.items()}
    with open(path, "w") as f:
        json.dump(serializable, f)

facenet_db = load_db("facenet_db.json")
live_json_path = "live_data.json"

# =========================
# Face Embedding
# =========================
def embed_face_aligned(frame, box):
    x1,y1,x2,y2 = map(int, box)
    h,w,_ = frame.shape
    margin = 30
    x1 = max(0, x1-margin); y1 = max(0, y1-margin)
    x2 = min(w, x2+margin); y2 = min(h, y2+margin)
    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0: return None
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    aligned = mtcnn(face_rgb)
    if aligned is None: return None
    aligned = aligned.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = facenet(aligned)[0]
        return F.normalize(emb, p=2, dim=0)

def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# =========================
# Register User
# =========================
def register_new_user(cap):
    global facenet_db
    name = input("Enter person's name: ").strip()
    if name in facenet_db:
        print("Name already exists!")
        return
    embeddings = []
    print("Capturing samples...")
    for _ in range(7):
        ret, frame = cap.read()
        if not ret: continue
        results = yolo_face_model(frame, conf=0.6, verbose=False)[0]
        if len(results.boxes) == 0: continue
        emb = embed_face_aligned(frame, results.boxes[0].xyxy[0])
        if emb is not None: embeddings.append(emb)
        time.sleep(0.4)
    if len(embeddings) < 3:
        print("Not enough samples.")
        return
    final_emb = F.normalize(torch.mean(torch.stack(embeddings), dim=0), p=2, dim=0)
    facenet_db[name] = final_emb
    save_db("facenet_db.json", facenet_db)
    print(f"{name} registered successfully!")

# =========================
# BLIP Caption Local
# =========================
def generate_scene_description(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = blip_model.generate(**inputs, max_new_tokens=25)
    return processor.decode(output[0], skip_special_tokens=True)

# =========================
# Camera
# =========================
cap = cv2.VideoCapture(0)
print("Press R to register | C for caption | Q to quit")

track_memory = {}
object_memory = set()
SIM_HISTORY = 5
THRESHOLD = 0.55

scene_desc = ""
last_scene_time = 0
scene_interval = 3  # كل 3 ثواني
last_saved_caption = ""

prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret: break
    curr_time = time.time()

    # ===== Face Detection =====
    results = yolo_face_model.track(frame, conf=0.6, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
    faces_live = []

    if results.boxes.id is not None:
        for box, track_id in zip(results.boxes, results.boxes.id):
            track_id = int(track_id)
            emb = embed_face_aligned(frame, box.xyxy[0])
            if emb is None: continue
            best_name = "Unknown"
            best_sim = -1
            for name, saved_emb in facenet_db.items():
                sim = cosine_sim(emb, saved_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
            if best_sim < THRESHOLD: best_name = "Unknown"

            if track_id not in track_memory:
                track_memory[track_id] = {"name_history": deque(maxlen=SIM_HISTORY)}

            track_memory[track_id]["name_history"].append(best_name)
            final_name = max(set(track_memory[track_id]["name_history"]),
                             key=track_memory[track_id]["name_history"].count)

            faces_live.append(final_name)

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            color = (0,255,0) if final_name != "Unknown" else (0,0,255)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,final_name,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    # ===== Object Detection =====
    detected_objects = []
    object_results = yolo_object_model(frame, conf=0.6, verbose=False)[0]
    for obj in object_results.boxes:
        obj_name = object_results.names[int(obj.cls)]
        if obj_name not in object_memory:
            object_memory.add(obj_name)
        detected_objects.append(obj_name)

    # ===== Scene Caption (Local BLIP) =====
    if curr_time - last_scene_time > scene_interval:
        new_caption = generate_scene_description(frame)
        if new_caption != last_saved_caption:
            scene_desc = new_caption
            last_saved_caption = new_caption
        last_scene_time = curr_time

    # ===== Live JSON (بدون تكرار غير ضروري) =====
    live_data = {
        "faces": list(set(faces_live)),
        "objects": list(set(detected_objects)),
        "scene_description": scene_desc,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(live_json_path, "w") as f:
        json.dump(live_data, f, indent=2)

    # ===== FPS =====
    delta_time = curr_time - prev_time if curr_time != prev_time else 0.001
    fps = 1/delta_time
    prev_time = curr_time
    cv2.putText(frame,f"FPS: {int(fps)}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    cv2.imshow("Face + Object + Scene Description (LOCAL)", frame)

    # ===== Keys =====
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        register_new_user(cap)
        track_memory.clear()
        object_memory.clear()
    elif key == ord('c'):
        print("Manual Caption:", generate_scene_description(frame))

cap.release()
cv2.destroyAllWindows()