import cv2
import yaml
import time
import os
import logging
from collections import deque
from vision.camera import Camera
from vision.detector import Detector
from vision.face_embedding import FaceEmbedder
from vision.emotion import EmotionClassifier
from memory.identity_store import IdentityStore
from memory.social_memory import SocialMemory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VisionTest")

def load_config(config_path="config/settings.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    # Initialize Modules (Matching AgentLoop setup)
    cfg = config
    camera = Camera(cfg['camera']['device_id'], cfg['camera']['width'], cfg['camera']['height'])
    detector = Detector(
        object_model_path=cfg['vision']['yolo_model'],
        face_model_path=cfg['vision'].get('yolo_face_model', "models/yolov8n-face.pt"),
        conf_threshold=cfg['vision']['conf_threshold']
    )
    
    face_embedder = FaceEmbedder(
        det_model_path=cfg['face'].get('det_model', "models/face_detection_yunet_2023mar.onnx"),
        rec_model_path=cfg['face'].get('rec_model', "models/face_recognition_sface_2021dec.onnx"),
        det_size=tuple(cfg['face'].get('det_size', [640, 640]))
    )
    
    emotion_classifier = EmotionClassifier()
    
    identity_store = IdentityStore(
        cfg['memory']['db_path'], 
        cfg['memory']['faiss_index'],
        embedding_dim=cfg['memory'].get('embedding_dim', 128)
    )
    social_memory = SocialMemory(cfg['memory']['db_path'])
    
    # Test State
    known_people = {} # track_id -> person_id
    vote_history = {} # track_id -> deque
    new_id_candidates = {} # track_id -> count of unknown detections
    
    os.makedirs("debug_vision", exist_ok=True)
    logger.info("Vision Modules Initialized. Starting live loop (press 'q' to exit)...")
    
    frame_count = 0
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                logger.warning("Failed to get frame")
                continue
                
            frame_count += 1
            
            # --- 1. PERCEIVE ---
            tracks = detector.detect_and_track(frame)
            
            h, w, _ = frame.shape
            
            for track in tracks:
                x1, y1, x2, y2 = track.box
                label = track.class_name
                color = (0, 255, 0)
                
                if track.class_name == 'person':
                    pid = known_people.get(track.track_id)
                    
                    if not pid or frame_count % 5 == 0:
                        face_res = face_embedder.extract(frame, box=(x1, y1, x2, y2))
                        if face_res:
                            frontal_score = face_res['frontal_score']
                            is_frontal = frontal_score > 0.7
                            
                            # 1. Try matching with ANY known view
                            matched_id, _ = identity_store.find_or_create(
                                face_res['embedding'], 
                                threshold=cfg['face']['similarity_threshold'], 
                                create=False
                            )
                            
                            if matched_id != "unknown":
                                pid = matched_id
                                # If we match a known person, but it's a side-view we haven't saved before,
                                # we could learn it here, but it's safer to only learn when track-locked.
                                logger.debug(f"Track {track.track_id} matched as {pid} (Frontal: {is_frontal})")
                            else:
                                # 2. If track IS ALREADY LOCKED but face didn't match (e.g. extreme side turn)
                                if pid and pid != "Identifying..." and pid != "unknown":
                                    # This is "Opportunistic Learning"
                                    # If it's a clear side-view, save it to their profile
                                    if frontal_score < 0.6: # Significant turn
                                        identity_store.find_or_create(
                                            face_res['embedding'], 
                                            person_id=pid,
                                            is_frontal=False
                                        )
                                        logger.info(f"Learned NEW SIDE VIEW for {pid}")
                                else:
                                    # 3. Discovery Phase: Only create NEW identities from frontal views
                                    new_id_candidates[track.track_id] = new_id_candidates.get(track.track_id, 0) + 1
                                    count = new_id_candidates[track.track_id]
                                    
                                    if is_frontal and count >= 5:
                                        pid, _ = identity_store.find_or_create(
                                            face_res['embedding'], 
                                            threshold=cfg['face']['similarity_threshold'], 
                                            create=True, 
                                            is_frontal=True
                                        )
                                        logger.info(f"Identity stable. Created: {pid}")
                                        del new_id_candidates[track.track_id]
                                    elif not is_frontal:
                                        logger.debug(f"Side view on track {track.track_id}, skipping discovery.")
                                        pid = pid or "Waiting for front view..."
                                    else:
                                        logger.info(f"Identifying {track.track_id}... {count}/5 (Frontal: {is_frontal})")
                                        pid = "Identifying..."
                            
                            if pid not in ["Identifying...", "unknown", "Waiting for front view..."]:
                                known_people[track.track_id] = pid
                                if track.track_id not in vote_history:
                                    vote_history[track.track_id] = deque([pid], maxlen=10)
                                else:
                                    vote_history[track.track_id].append(pid)
                                
                                # 4. Emotion (Only if identified)
                                if frame_count % cfg['vision']['emotion_every_n_frames'] == 0:
                                    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                                    if crop.size > 0:
                                        emo_probs = emotion_classifier.predict(crop)
                                        top_emo = max(emo_probs, key=emo_probs.get)
                                        label += f" | {top_emo}"
                        else:
                            pid = known_people.get(track.track_id, "unknown")
                    
                    # Voting logic: Use most common ID in history
                    if track.track_id in vote_history and vote_history[track.track_id]:
                        final_pid = max(set(vote_history[track.track_id]), key=list(vote_history[track.track_id]).count)
                    else:
                        final_pid = pid
                    
                    mem = social_memory.get(final_pid)
                    name = mem.get('name', final_pid)
                    label = f"{name} [ID:{track.track_id}]"
                    color = (255, 0, 0)

                # Draw Overlay
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Show Live Preview
            cv2.imshow("Vision Test - Press 'q' to Exit", frame)
            
            # Exit on key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        logger.info(f"Test complete.")

if __name__ == "__main__":
    main()
