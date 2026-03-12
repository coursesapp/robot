import cv2
import torch
import time
import json
import os
import numpy as np
import threading
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch.nn.functional as F
from collections import deque
from queue import Queue, Empty

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Models
# =========================
yolo_face_model = YOLO("yolov8n-face.pt")
yolo_object_model = YOLO("yolov8n.pt")

mtcnn = MTCNN(image_size=160, margin=20, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

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

# =========================
# Shared Variables
# =========================
frame_queue = Queue(maxsize=1)  # آخر فريم فقط يخزن
processed_frame = None
result_lock = threading.Lock()
running = True

track_memory = {}
detected_object_ids = {}
SIM_HISTORY = 5
THRESHOLD = 0.55

os.makedirs("objects", exist_ok=True)

# =========================
# Embedding
# =========================
def embed_face_aligned(frame, box):
    x1,y1,x2,y2 = map(int, box)
    h,w,_ = frame.shape
    margin = 30
    x1 = max(0, x1-margin)
    y1 = max(0, y1-margin)
    x2 = min(w, x2+margin)
    y2 = min(h, y2+margin)

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None

    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    aligned = mtcnn(face_rgb)
    if aligned is None:
        return None

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
        if not ret:
            continue
        results = yolo_face_model(frame, conf=0.6, verbose=False)[0]
        if len(results.boxes) == 0:
            continue
        emb = embed_face_aligned(frame, results.boxes[0].xyxy[0])
        if emb is not None:
            embeddings.append(emb)
        time.sleep(0.4)
    if len(embeddings) < 3:
        print("Failed to capture enough samples.")
        return
    final_emb = torch.mean(torch.stack(embeddings), dim=0)
    final_emb = F.normalize(final_emb, p=2, dim=0)
    facenet_db[name] = final_emb
    save_db("facenet_db.json", facenet_db)
    print(f"{name} registered successfully!")

# =========================
# Camera Thread
# =========================
def camera_thread():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    global running
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        # ضع آخر فريم فقط
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except Empty:
                pass
        frame_queue.put(frame.copy())

    cap.release()

# =========================
# Processing Thread
# =========================
def processing_thread():
    global processed_frame
    prev_time = time.time()

    while running:
        try:
            frame = frame_queue.get(timeout=0.1)
        except Empty:
            continue
        # ================= FACE =================
        results = yolo_face_model.track(
            frame,
            conf=0.6,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )[0]

        if results.boxes.id is not None:
            for box, track_id in zip(results.boxes, results.boxes.id):
                track_id = int(track_id)
                emb = embed_face_aligned(frame, box.xyxy[0])
                if emb is None:
                    continue

                best_name = "Unknown"
                best_sim = -1
                for name, saved_emb in facenet_db.items():
                    sim = cosine_sim(emb, saved_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name
                if best_sim < THRESHOLD:
                    best_name = "Unknown"

                if track_id not in track_memory:
                    track_memory[track_id] = {
                        "name_history": deque(maxlen=SIM_HISTORY),
                        "sim_history": deque(maxlen=SIM_HISTORY)
                    }

                track_memory[track_id]["name_history"].append(best_name)
                track_memory[track_id]["sim_history"].append(best_sim)

                final_name = max(
                    set(track_memory[track_id]["name_history"]),
                    key=track_memory[track_id]["name_history"].count
                )

                avg_sim = np.mean(track_memory[track_id]["sim_history"])
                confidence = int(avg_sim*100)

                x1,y1,x2,y2 = map(int, box.xyxy[0])
                label = f"{final_name} ({confidence}%)"
                color = (0,255,0) if final_name != "Unknown" else (0,0,255)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

        # ================= OBJECT =================
        object_results = yolo_object_model.track(
            frame,
            conf=0.5,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )[0]

        current_objects = []
        if object_results.boxes.id is not None:
            for box, obj_id in zip(object_results.boxes, object_results.boxes.id):
                obj_id = int(obj_id)
                cls_id = int(box.cls[0])
                class_name = yolo_object_model.names[cls_id]
                conf = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]

                if obj_id not in detected_object_ids:
                    detected_object_ids[obj_id] = {
                        "class": class_name,
                        "first_seen": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    if crop.size != 0:
                        cv2.imwrite(f"objects/{class_name}_{obj_id}.jpg", crop)

                current_objects.append({
                    "id": obj_id,
                    "class": class_name,
                    "confidence": int(conf*100),
                    "first_seen": detected_object_ids[obj_id]["first_seen"]
                })

                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)
                cv2.putText(frame,
                            f"{class_name} ID:{obj_id}",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255,255,0),
                            2)

        # ================= JSON =================
        current_faces = []
        for track_id, data in track_memory.items():
            final_name = max(set(data["name_history"]),
                             key=data["name_history"].count)
            avg_sim = np.mean(data["sim_history"])
            confidence = int(avg_sim*100)
            current_faces.append({
                "id": track_id,
                "name": final_name,
                "confidence": confidence
            })

        live_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "faces_detected_now": current_faces,
            "objects_detected_now": current_objects,
            "total_unique_objects_seen": len(detected_object_ids)
        }

        with open("live_output.json", "w") as f:
            json.dump(live_data, f, indent=4)

        # ================= FPS =================
        curr_time = time.time()
        fps = 1/(curr_time-prev_time)
        prev_time = curr_time
        cv2.putText(frame,f"FPS: {int(fps)}",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

        with result_lock:
            processed_frame = frame.copy()

# =========================
# Start Threads
# =========================
cam_thread = threading.Thread(target=camera_thread)
proc_thread = threading.Thread(target=processing_thread)

cam_thread.start()
proc_thread.start()

# =========================
# Display Loop
# =========================
while True:
    with result_lock:
        if processed_frame is not None:
            cv2.imshow("Alzheimer Assist System", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break
    elif cv2.waitKey(1) & 0xFF == ord('r'):
        # استخدم أحدث فريم للتسجيل
        with frame_queue.mutex:
            if frame_queue.queue:
                latest_for_register = frame_queue.queue[-1]
                register_new_user(cv2.VideoCapture(0))

cv2.destroyAllWindows()
cam_thread.join()
proc_thread.join()