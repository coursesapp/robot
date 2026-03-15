import cv2
import time
import yaml
import numpy as np
import threading
from collections import deque
from pathlib import Path

from vision.camera import Camera
from vision.detector import Detector
from vision.face_embedding import FaceEmbedder
from audio.stt import STTEngine
from audio.speaker_id import SpeakerRecognizer
from memory.identity_store import IdentityStore

def load_config(path="config/settings.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class FusionTester:
    def __init__(self):
        self.config = load_config()
        cfg = self.config
        
        # Initialize Vision
        self.camera = Camera(cfg['camera']['device_id'], cfg['camera']['width'], cfg['camera']['height'])
        self.detector = Detector(
            object_model_path=cfg['vision']['yolo_model'],
            face_model_path=cfg['vision'].get('yolo_face_model', "models/yolov8n-face.pt"),
            conf_threshold=cfg['vision']['conf_threshold']
        )
        self.face_embedder = FaceEmbedder(
            det_model_path=cfg['face'].get('det_model', "models/face_detection_yunet_2023mar.onnx"),
            rec_model_path=cfg['face'].get('rec_model', "models/face_recognition_sface_2021dec.onnx")
        )

        # Initialize Memory
        self.identity_store = IdentityStore(
            cfg['memory']['db_path'], 
            cfg['memory']['faiss_index'],
            embedding_dim=cfg['memory'].get('embedding_dim', 128)
        )

        # Initialize Audio
        self.speaker_id = SpeakerRecognizer(cfg['audio'].get('speaker_model', "models/voxceleb_CAM++.onnx"))
        self.stt = STTEngine(cfg['audio']['whisper_model'], callback=self.on_speech)
        
        # State
        self.mouth_history = {} # track_id -> deque
        self.known_people = {} # track_id -> pid
        self.last_speech = ""
        self.last_speaker_pid = "None"
        self.frame_count = 0
        self.running = True

    def on_speech(self, text, audio_np):
        print(f"\n[STT] Heard: {text}")
        
        # 1. Correlate with Visual Activity FIRST
        best_track_id = None
        max_activity = 0
        for tid, history in self.mouth_history.items():
            if len(history) > 5:
                activity = np.std(history)
                if activity > max_activity:
                    max_activity = activity
                    best_track_id = tid
        
        visual_pid = self.known_people.get(best_track_id, "unknown") if best_track_id else "unknown"

        # 2. Identify Speaker by Voice (and LEARN if visual_pid is known)
        voice_pid = "unknown"
        voice_embedding = self.speaker_id.extract_embedding(audio_np)
        if voice_embedding is not None:
            # We pass visual_pid so the store can link the voice if it's new
            voice_pid, _ = self.identity_store.find_or_create_voice(voice_embedding, person_id=visual_pid)
            print(f"[Speaker ID] Voice matched/linked to: {voice_pid}")

        # 3. Fusion Logic
        final_pid = voice_pid
        if voice_pid == "unknown":
            final_pid = visual_pid
        elif best_track_id and visual_pid == "unknown":
            # Label the unknown face based on voice recognition result!
            print(f"[Fusion] Labeling unknown track {best_track_id} as {voice_pid} via voice recognition.")
            self.known_people[best_track_id] = voice_pid

        self.last_speech = text
        self.last_speaker_pid = final_pid
        print(f"[Fusion Result] Decided Speaker: {final_pid}\n")

    def run(self):
        print("Starting Fusion Test... Press 'q' to quit.")
        self.stt.start()
        
        id_freq = self.config['vision'].get('identification_every_n_frames', 10)
        
        try:
            while self.running:
                frame = self.camera.get_frame()
                if frame is None: continue
                
                self.frame_count += 1
                tracks = self.detector.detect_and_track(frame)
                
                for track in tracks:
                    if track.class_name == 'person':
                        # 1. Mouth Tracking (Every frame)
                        x1, y1, x2, y2 = track.box
                        should_recognize = (track.track_id not in self.known_people or self.frame_count % id_freq == 0)
                        
                        face_res = self.face_embedder.extract(frame, box=(x1, y1, x2, y2), extract_embedding=should_recognize)
                        
                        if face_res:
                            # Update mouth history
                            if 'mouth_dist' in face_res:
                                if track.track_id not in self.mouth_history:
                                    self.mouth_history[track.track_id] = deque(maxlen=20)
                                self.mouth_history[track.track_id].append(face_res['mouth_dist'])
                            
                            # Update identity
                            if should_recognize and face_res.get('embedding') is not None:
                                pid, _ = self.identity_store.find_or_create(
                                    face_res['embedding'], 
                                    threshold=self.config['face']['similarity_threshold'],
                                    create=True # Ensure we create persona-X if unknown
                                )
                                if pid != "unknown":
                                    self.known_people[track.track_id] = pid
                                    print(f"[Vision ID] Person detected: {pid} on track {track.track_id}")
                        
                        # Draw Info
                        pid = self.known_people.get(track.track_id, "Unknown")
                        color = (0, 255, 0) if pid == self.last_speaker_pid else (255, 0, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"ID: {pid}"
                        if track.track_id in self.mouth_history and len(self.mouth_history[track.track_id]) > 0:
                            act = np.std(self.mouth_history[track.track_id])
                            label += f" | Act: {act:.4f}"
                        
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Overlay Last Speech
                cv2.putText(frame, f"Last Speaker: {self.last_speaker_pid}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Heard: {self.last_speech}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow("Audio-Visual Fusion Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.stt.stop()
            self.camera.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    tester = FusionTester()
    tester.run()
