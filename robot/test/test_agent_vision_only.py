import yaml
import logging
import cv2
import time
from core.agent_loop import AgentLoop

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VisionOnlyTest")

def main():
    # Load config
    with open("config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize AgentLoop
    agent = AgentLoop(config)
    
    # --- PARTIAL START (Vision Only) ---
    logger.info("Initializing Vision-Only Mode...")
    agent.cfg = config
    
    # 1. Start Camera
    from vision.camera import Camera
    agent.camera = Camera(config['camera']['device_id'], config['camera']['width'], config['camera']['height'])
    
    # 2. Start Detector
    from vision.detector import Detector
    agent.detector = Detector(
        object_model_path=config['vision']['yolo_model'],
        face_model_path=config['vision'].get('yolo_face_model', "models/yolov8n-face.pt"),
        conf_threshold=config['vision']['conf_threshold']
    )
    
    # 3. Start Face Embedder
    from vision.face_embedding import FaceEmbedder
    agent.face_embedder = FaceEmbedder(
        det_model_path=config['face'].get('det_model', "models/face_detection_yunet_2023mar.onnx"),
        rec_model_path=config['face'].get('rec_model', "models/face_recognition_sface_2021dec.onnx"),
        det_size=tuple(config['face'].get('det_size', [640, 640]))
    )
    
    # 4. Start Memory
    from memory.identity_store import IdentityStore
    from memory.social_memory import SocialMemory
    agent.identity_store = IdentityStore(
        config['memory']['db_path'], 
        config['memory']['faiss_index'],
        embedding_dim=config['memory'].get('embedding_dim', 128)
    )
    agent.social_memory = SocialMemory(config['memory']['db_path'])
    
    # Optional: Dashboard (useful for seeing results)
    agent.dashboard.start()
    
    # Start the Vision Thread (Producer) manually for this test
    import threading
    agent.running = True
    agent.vision_thread = threading.Thread(target=agent._vision_worker, daemon=True)
    agent.vision_thread.start()
    
    logger.info("Vision-Only Mode Ready. Starting Loop...")
    
    try:
        while True:
            # Consume from the agent's internal vision thread
            with agent.vision_lock:
                frame = agent.latest_frame
                tracks = list(agent.latest_tracks)
            
            if frame is None:
                time.sleep(0.01)
                continue
                
            frame_display = frame.copy()
            agent.frame_count += 1
            
            # Run identity logic (Simulating the 'Brain' Consumer)
            for track in tracks:
                if track.class_name == 'person':
                    pid = agent.known_people.get(track.track_id)
                    
                    # Periodic Recognition Verification (The "Brain" part)
                    if not pid or agent.frame_count % 10 == 0:
                        x1, y1, x2, y2 = track.box
                        face_res = agent.face_embedder.extract(frame, box=(x1, y1, x2, y2))
                        
                        if face_res:
                            # Use exact logic from agent_loop.py
                            fs = face_res['frontal_score']
                            is_frontal = fs > 0.7
                            threshold = config['face']['similarity_threshold']
                            
                            matched_id, _ = agent.identity_store.find_or_create(face_res['embedding'], threshold=threshold, create=False)
                            
                            if matched_id != "unknown":
                                pid = matched_id
                            else:
                                if pid and pid not in ["unknown", "Identifying...", "Waiting for front view..."]:
                                    if fs < 0.6:
                                        agent.identity_store.find_or_create(face_res['embedding'], person_id=pid, is_frontal=False)
                                        logger.info(f"Learned side-view for {pid}")
                                else:
                                    count = agent.new_id_candidates.get(track.track_id, 0) + 1
                                    agent.new_id_candidates[track.track_id] = count
                                    if is_frontal and count >= 5:
                                        pid, _ = agent.identity_store.find_or_create(face_res['embedding'], threshold=threshold, create=True, is_frontal=True)
                                        del agent.new_id_candidates[track.track_id]
                                    elif not is_frontal:
                                        pid = "Waiting for front view..."
                                    else:
                                        pid = "Identifying..."
                            
                            if pid not in ["unknown", "Identifying...", "Waiting for front view..."]:
                                agent.known_people[track.track_id] = pid

                    # Display Logic
                    label = f"{pid or 'Identifying...'} [ID:{track.track_id}]"
                    x1, y1, x2, y2 = track.box
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("AgentLoop - Threaded Vision Test", frame_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        agent.camera.stop()
        agent.dashboard.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
