import time
import numpy as np
import logging
import threading
import queue
from typing import Dict, Any, Optional
from collections import deque
import os
import cv2

from core.event_bus import EventBus
from vision.camera import Camera
from vision.detector import Detector
from vision.face_embedding import FaceEmbedder
from vision.emotion import EmotionClassifier
from audio.stt import STTEngine
from audio.tts import TTSEngine
from audio.speaker_id import SpeakerRecognizer
from dialogue.llm_client import LLMClient
from dialogue.prompt_engine import PromptEngine
from dialogue.info_extractor import InfoExtractor
from dialogue.summarizer import Summarizer
from memory.identity_store import IdentityStore
from memory.social_memory import SocialMemory
from memory.vector_memory import VectorMemory
from dashboard.web_dashboard import WebDashboard
from core.actions import ActionLibrary
from core.interaction import InteractionStrategy
from core.local_strategy import LocalStrategy
import random
import json

logger = logging.getLogger("AgentLoop")

class AgentLoop:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bus = EventBus()
        self.running = False
        self.stop_event = threading.Event()
        
        # Modules
        self.camera = None
        self.detector = None
        self.tracker = None
        self.face_embedder = None
        self.emotion_classifier = None
        self.stt = None
        self.tts = None
        self.llm = None
        self.prompt_engine = None
        self.identity_store = None
        self.social_memory = None
        self.dashboard = WebDashboard(port=5050)
        
        # Loop state
        self.fps = config['camera'].get('fps', 15)
        self.frame_interval = 1.0 / self.fps
        self.last_frame_time = 0
        self.frame_count = 0
        self.llm_busy = False  # Prevent overlapping LLM calls

        # Threading state
        self.latest_frame = None
        self.latest_tracks = []
        self.vision_lock = threading.Lock()
        self.vision_thread = None
        
        # Context state
        self.current_speech = queue.Queue()
        self.known_people = {} # track_id -> person_id (Loyal Track Lock)
        self.vote_history = {} # track_id -> deque(maxlen=10)
        self.new_id_candidates = {} # track_id -> count of unknown detections
        self.context_history = [] # List[str]
        self.seen_object_ids = set() # set of track_id
        self.last_seen_ids = {} # track_id -> timestamp (For GC)
        self.last_greeted = {} # person_id -> timestamp
        self.last_thought = "None"
        
        # Audio-Visual Fusion State
        self.speaker_id_engine = None
        self.mouth_history = {} # track_id -> deque(maxlen=20)
        
        # Interaction Strategy
        self.interaction_strategy: Optional[InteractionStrategy] = None

    def start(self):
        """Initialize all modules."""
        logger.info("Initializing Agent Modules...")
        cfg = self.config
        self.cfg = cfg  # Store for access from worker threads

        # Vision
        self.camera = Camera(cfg['camera']['device_id'], cfg['camera']['width'], cfg['camera']['height'])
        self.detector = Detector(
            object_model_path=cfg['vision']['yolo_model'],
            face_model_path=cfg['vision'].get('yolo_face_model', "models/yolov8n-face.pt"),
            conf_threshold=cfg['vision']['conf_threshold']
        )
        
        # Update: Use OpenCV Face models
        self.face_embedder = FaceEmbedder(
            det_model_path=cfg['face'].get('det_model', "models/face_detection_yunet_2023mar.onnx"),
            rec_model_path=cfg['face'].get('rec_model', "models/face_recognition_sface_2021dec.onnx"),
            det_size=tuple(cfg['face'].get('det_size', [640, 640]))
        )
        
        self.emotion_classifier = EmotionClassifier() # optional path

        # Memory (Update: use correct embedding dim)
        self.identity_store = IdentityStore(
            cfg['memory']['db_path'], 
            cfg['memory']['faiss_index'],
            embedding_dim=cfg['memory'].get('embedding_dim', 128)
        )
        self.social_memory = SocialMemory(cfg['memory']['db_path'])
        
        # Long-Term Hybrid Vector Memory (ChromaDB)
        self.vector_memory = VectorMemory(cfg['memory'].get('chroma_db_path', 'memory/chroma_db'))

        # Audio
        self.stt = STTEngine(cfg['audio']['whisper_model'], 
                            callback=self.on_speech, 
                            on_speech_start=self.on_speech_start,
                            on_processing=self.on_speech_processing)
        self.stt.start()
        
        self.tts = TTSEngine(cfg['audio']['piper_model'], 
                            cfg['audio'].get('piper_config'),
                            on_start=self.on_tts_start,
                            on_end=self.on_tts_end)
        self.tts.start()

        self.speaker_id_engine = SpeakerRecognizer(cfg['audio'].get('speaker_model', "models/voxceleb_CAM++.onnx"))

        # Dialogue (Supports Ollama or llama-server via config)
        self.llm = LLMClient(cfg.get('llm', {}))
        self.prompt_engine = PromptEngine()
        self.info_extractor = InfoExtractor(self.llm)
        self.summarizer = Summarizer(self.llm)
        self.action_library = ActionLibrary()
        
        # State for proactive behavior
        self.last_greeted = {} # track_id -> timestamp
        self.visual_conf = {}  # track_id -> confidence score
        self.turn_count = 0
        
        # Dashboard
        self.dashboard.start()

        # Start Vision Thread (Producer)
        self.running = True
        self.vision_thread = threading.Thread(target=self._vision_worker, daemon=True)
        self.vision_thread.start()

        logger.info("All modules initialized.")

    def _vision_worker(self):
        """Fast thread for camera and YOLO tracking."""
        logger.info(f"Vision Worker Thread Started (Limited to {self.fps} FPS).")
        worker_interval = 1.0 / self.fps
        
        while self.running or not self.stop_event.is_set():
            worker_start = time.time()
            if not self.camera:
                time.sleep(0.1)
                continue
                
            frame = self.camera.get_frame()
            if frame is None:
                continue

            # Run tracking
            tracks = self.detector.detect_and_track(frame)
            
            # Tracks in current frame
            current_tids = {t.track_id for t in tracks}
            
            with self.vision_lock:
                self.latest_frame = frame
                self.latest_tracks = tracks # for LLM
                
                # Cleanup mouth history for tracks NOT in current frame
                for tid in list(self.mouth_history.keys()):
                    if tid not in current_tids:
                        self.mouth_history.pop(tid, None)
            
            # Enforce FPS worker limit to save CPU
            elapsed = time.time() - worker_start
            if elapsed < worker_interval:
                time.sleep(worker_interval - elapsed)

    def on_speech_start(self):
        """Instant callback when user starts talking (for local mute)."""
        self.dashboard.update_state(lambda st: setattr(st, 'avatar_state', 'greeting'))
        if self.interaction_strategy:
            self.interaction_strategy.on_speech_start()

    def on_speech_processing(self):
        """Called when user stops talking and STT starts transcribing."""
        self.dashboard.update_state(lambda st: setattr(st, 'avatar_state', 'thinking'))

    def on_tts_start(self):
        """Called when agent starts speaking."""
        self.dashboard.update_state(lambda st: setattr(st, 'avatar_state', 'explaining'))

    def on_tts_end(self):
        """Called when agent finishes speaking."""
        # Only reset to idle if no other high-priority states are active
        self.dashboard.update_state(lambda st: setattr(st, 'avatar_state', 'idle'))

    def on_speech(self, text: str, audio_np: np.ndarray):
        """
        Callback from STT when user finishes speaking.
        Includes the raw audio for Speaker Identification.
        """
        logger.info(f"STT Callback: '{text}'")
        
        # 1. Correlate with Visual Mouth Movement FIRST
        best_track_id = None
        now_t = time.time()
        with self.vision_lock:
            max_activity = 0
            for tid, history in self.mouth_history.items():
                # ONLY consider tracks seen in the last 0.5 seconds (effectively currently visible)
                last_seen = self.last_seen_ids.get(tid, 0)
                if (now_t - last_seen) < 0.5 and len(history) > 5:
                    # activity defined as standard deviation of mouth distance
                    activity = np.std(history)
                    if activity > max_activity:
                        max_activity = activity
                        best_track_id = tid
        
        visual_pid = self.known_people.get(best_track_id, "unknown") if best_track_id else "unknown"
        visual_conf = self.visual_conf.get(best_track_id, 0.0) if best_track_id else 0.0

        # 2. Identify Speaker by Voice
        voice_pid = "unknown"
        voice_conf = 0.0
        if self.speaker_id_engine:
            voice_embedding = self.speaker_id_engine.extract_embedding(audio_np)
            if voice_embedding is not None:
                voice_pid, _, voice_conf = self.identity_store.find_or_create_voice(voice_embedding, person_id=visual_pid)
                logger.info(f"Voice Identity: {voice_pid} (conf: {voice_conf:.2f})")

        # 3. Audio-First Fusion & Conflict Detection
        final_pid = "unknown"
        identity_conflict = False
        conflict_msg = ""

        # Logic: Priority for Audio if it's highly confident
        # We only trigger a conflict if BOTH are very confident and disagree.
        if voice_conf > 0.8:
            final_pid = voice_pid
            if visual_pid != "unknown" and visual_pid != voice_pid and visual_conf > 0.85:
                identity_conflict = True
                voice_name = self.social_memory.get(voice_pid).get('name', voice_pid)
                visual_name = self.social_memory.get(visual_pid).get('name', visual_pid)
                conflict_msg = f"Identity Conflict: Voice is strongly {voice_name}, but Vision is certain it sees {visual_name} speaking."
                logger.warning(conflict_msg)
        elif visual_conf > 0.7:
            # Fallback to visual ID if audio is not enough
            final_pid = visual_pid
        elif voice_conf > 0.6:
            # If vision is low, but audio is decent, trust audio
            final_pid = voice_pid
        else:
            final_pid = visual_pid # Final fallback to visual track

        # 4. Delegate to Active Interaction Strategy
        metadata = {
            "final_pid": final_pid,
            "current_emotions": self.dashboard.state.emotions if hasattr(self.dashboard.state, 'emotions') else [],
            "tracks": self.latest_tracks,
            "wh": (self.camera.width, self.camera.height),
            "identity_conflict": identity_conflict,
            "conflict_message": conflict_msg
        }
        self.interaction_strategy.on_speech(text, audio_np, metadata)
        
        # Update Dashboard
        def update_stt(state):
            mem = self.social_memory.get(final_pid) if final_pid != "unknown" else {}
            display_name = mem.get('name') or final_pid
            # Log confidence scores for debugging
            conf_str = f" [👁️{visual_conf:.2f} 🎙️{voice_conf:.2f}]"
            state.add_event(f"Heard ({display_name}){conf_str}: {text}")
        self.dashboard.update_state(update_stt)

    def stop(self):
        """Graceful shutdown."""
        self.running = False
        self.stop_event.set()
        
        if self.camera: self.camera.stop()
        if self.stt: self.stt.stop()
        if self.tts: self.tts.stop()
        if self.dashboard: self.dashboard.stop()
            
        logger.info("Agent Loop stopped.")

    def run(self, duration: int = 0):
        """Main blocking loop."""
        self.start()
        
        # Interaction Strategy Initialization
        mode = self.config.get('interaction', {}).get('mode', 'local')
        if mode == 'gemini_live':
            # Placeholder for now, we'll implement this next
            try:
                from core.gemini_strategy import GeminiLiveStrategy
                self.interaction_strategy = GeminiLiveStrategy(self)
            except ImportError:
                logger.warning("GeminiLiveStrategy not found, falling back to LocalStrategy")
                self.interaction_strategy = LocalStrategy(self)
        else:
            self.interaction_strategy = LocalStrategy(self)

        self.running = True
        start_time = time.time()
        
        try:
            while self.running:
                now = time.time()
                
                if duration > 0 and (now - start_time) > duration:
                    logger.info("Duration limit reached.")
                    break
                
                # Enforce FPS
                elapsed = now - self.last_frame_time
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)
                    continue
                
                self.last_frame_time = now
                self.frame_count += 1
                
                # --- 1. PERCEIVE (Consumer) ---
                with self.vision_lock:
                    if self.latest_frame is None:
                        continue
                    frame = self.latest_frame.copy()
                    tracks = list(self.latest_tracks)
                
                h, w, _ = frame.shape
                current_identities = []
                current_emotions = []
                
                for track in tracks:
                    self.last_seen_ids[track.track_id] = now
                    
                    if track.class_name == 'person':
                        pid = self.known_people.get(track.track_id)
                        x1, y1, x2, y2 = track.box

                        # --- Dual-Frequency Perception ---
                        # We track mouth and embedding based on config frequencies
                        id_freq = self.config['vision'].get('identification_every_n_frames', 10)
                        mouth_freq = self.config['vision'].get('mouth_track_every_n_frames', 1)
                        
                        should_recognize = (not pid or self.frame_count % id_freq == 0)
                        should_track_mouth = (self.frame_count % mouth_freq == 0)
                        
                        # We only need landmarks if we are tracking mouth or recognizing
                        if should_recognize or should_track_mouth:
                            face_res = self.face_embedder.extract(
                                frame, 
                                box=(x1, y1, x2, y2), 
                                extract_embedding=should_recognize
                            )
                        else:
                            face_res = None
                        
                        if face_res:
                            # 1. Mouth Activity Tracking (Always available in face_res if called)
                            if should_track_mouth and 'mouth_dist' in face_res:
                                if track.track_id not in self.mouth_history:
                                    self.mouth_history[track.track_id] = deque(maxlen=20)
                                self.mouth_history[track.track_id].append(face_res['mouth_dist'])

                            # 2. Identity Matching (Only if we requested embedding)
                            if should_recognize and face_res.get('embedding') is not None:
                                fs = face_res['frontal_score']
                                is_frontal = fs > 0.7
                                threshold = self.config['face']['similarity_threshold']
                                
                                # Identity Matching
                                matched_id, _, visual_conf = self.identity_store.find_or_create(face_res['embedding'], threshold=threshold, create=False)
                                
                                if matched_id != "unknown":
                                    # Identity Conflict Check: If track was labeled differently, reset it
                                    if pid != "unknown" and pid != "Identifying..." and pid != matched_id:
                                        logger.warning(f"TRACK CONFLICT: Track {track.track_id} was {pid}, now recognized as {matched_id}. Resetting identity.")
                                        self.mouth_history.pop(track.track_id, None)
                                        self.vote_history.pop(track.track_id, None)
                                        self.known_people[track.track_id] = matched_id
                                    
                                    pid = matched_id
                                    self.visual_conf[track.track_id] = visual_conf
                                else:
                                    if pid and pid not in ["unknown", "Identifying...", "Waiting for front view..."]:
                                        if not is_frontal:
                                            self.identity_store.find_or_create(face_res['embedding'], person_id=pid, is_frontal=False)
                                    else:
                                        # Count frames to ensure stable face before creating new persona
                                        count = self.new_id_candidates.get(track.track_id, 0) + 1
                                        self.new_id_candidates[track.track_id] = count
                                        
                                        # If we have 3 stable frames (even better if frontal), create ID
                                        if (is_frontal and count >= 5) or count >= 10:
                                            pid, _, visual_conf = self.identity_store.find_or_create(
                                                face_res['embedding'], 
                                                threshold=threshold, 
                                                create=True, 
                                                is_frontal=is_frontal
                                            )
                                            self.visual_conf[track.track_id] = visual_conf
                                            del self.new_id_candidates[track.track_id]
                                            
                                if pid not in ["unknown", "Identifying...", "Waiting for front view..."]:
                                    self.known_people[track.track_id] = pid
                                    if track.track_id not in self.vote_history:
                                        self.vote_history[track.track_id] = deque([pid] * 5, maxlen=10)
                                    else:
                                        self.vote_history[track.track_id].append(pid)
                                    
                                    # Emotion
                                    if self.frame_count % self.config['vision']['emotion_every_n_frames'] == 0:
                                        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                                        if crop.size > 0:
                                            emo_probs = self.emotion_classifier.predict(crop)
                                            top_emo = max(emo_probs, key=emo_probs.get)
                                            current_emotions.append(top_emo)
                        
                        # Apply Identity Voting
                        if track.track_id in self.vote_history:
                            final_pid = max(set(self.vote_history[track.track_id]), key=list(self.vote_history[track.track_id]).count)
                        else:
                            final_pid = pid or "unknown"
                            
                        if self.known_people.get(track.track_id) != final_pid:
                            self.known_people[track.track_id] = final_pid
                            if final_pid != "unknown":
                                soc_name = self.social_memory.get(final_pid).get('name')
                                disp = soc_name if soc_name else final_pid
                                self.dashboard.update_state(lambda st, d=disp: st.add_event(f"✅ Identity Settled: {d}"))

                        if final_pid and final_pid != "unknown":
                            current_identities.append(final_pid)
                            
                            # Proactive Greeting logic
                            now_t = time.time()
                            last_g_time = self.last_greeted.get(final_pid, 0)
                            last_any_greet = self.last_greeted.get("_GLOBAL_", 0)
                            cooldown = self.config.get('agent', {}).get('greet_cooldown', 120)
                            
                            if track.track_id in self.vote_history and len(self.vote_history[track.track_id]) >= 5 and (now_t - last_g_time > cooldown) and (now_t - last_any_greet > 20):
                                has_name = bool(self.social_memory.get(final_pid).get('name'))
                                event_text = ""
                                if not has_name:
                                    event_text = f"[System Event: You just saw a person (ID: {final_pid}) whose name you don't know yet. Greet them warmly and ask for their name.]"
                                else:
                                    if random.random() < self.config.get('agent', {}).get('sociality', 0.7):
                                        event_text = f"[System Event: You just saw a known person (ID: {final_pid}). Start a friendly conversation.]"
                                
                                if event_text:
                                    # Pass to strategy as a system event (no audio)
                                    self.dashboard.update_state(lambda st: setattr(st, 'avatar_state', 'greeting'))
                                    self.interaction_strategy.on_speech(event_text, None, {
                                        "final_pid": final_pid, 
                                        "current_emotions": current_emotions,
                                        "is_system_event": True
                                    })
                                    self.last_greeted[final_pid] = now_t
                                    self.last_greeted["_GLOBAL_"] = now_t
                    else:
                        # Non-person Object Snapshot
                        if track.track_id not in self.seen_object_ids:
                            self.seen_object_ids.add(track.track_id)
                            x1, y1, x2, y2 = track.box
                            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                            if crop.size > 0:
                                os.makedirs("memory/object_snapshots", exist_ok=True)
                                cv2.imwrite(f"memory/object_snapshots/{track.class_name}_{track.track_id}.jpg", crop)

                # 3. Update Interaction Strategy with Vision Data
                self.interaction_strategy.on_vision(frame, {
                    "tracks": tracks,
                    "wh": (w, h),
                    "emotions": current_emotions
                })

                # --- 4. GARBAGE COLLECTION ---
                if self.frame_count % 100 == 0:
                    now_t = time.time()
                    # Global track cleanup (30s timeout)
                    stale_ids = [tid for tid, lt in self.last_seen_ids.items() if now_t - lt > 30.0]
                    # Aggressive mouth history cleanup (5s timeout)
                    mouth_stale = [tid for tid in list(self.mouth_history.keys()) if now_t - self.last_seen_ids.get(tid, 0) > 5.0]
                    
                    for tid in mouth_stale:
                        self.mouth_history.pop(tid, None)

                    for tid in stale_ids:
                        self.known_people.pop(tid, None)
                        self.vote_history.pop(tid, None)
                        self.new_id_candidates.pop(tid, None)
                        self.last_seen_ids.pop(tid, None)
                        self.mouth_history.pop(tid, None)
                        self.visual_conf.pop(tid, None)
                    if stale_ids:
                        logger.debug(f"GC: Cleaned up {len(stale_ids)} stale track IDs.")

                # --- Update Dashboard ---
                display_names = {}
                for tid, p in self.known_people.items():
                    mem = self.social_memory.get(p) if p != "unknown" else {}
                    display_names[tid] = {
                        "name": mem.get('name', p),
                        "emotion": "",
                        "job": mem.get('job', ""),
                        "interests": mem.get('interests', []),
                        "is_known_pid": p != "unknown"
                    }
                
                dash_objects = []
                for t in tracks:
                    if t.class_name != 'person':
                        cx = (t.box[0] + t.box[2]) / 2.0
                        pos = "left" if cx < w/3 else "right" if cx > 2*w/3 else "center"
                        dash_objects.append({"label": t.class_name, "position": pos})

                self.dashboard.update_state(lambda st, dn=display_names, objs=dash_objects, fr=frame: 
                    setattr(st, 'camera_fps', self.fps) or 
                    setattr(st, 'known_people', dn) or 
                    setattr(st, 'objects', objs) or 
                    setattr(st, 'frame', fr))

                # --- Live JSON Output ---
                try:
                    live_faces = [{"id": tid, "name": info['name']} for tid, info in display_names.items() if info['name'] != "unknown"]
                    live_objs = [{"label": obj['label'], "position": obj['position']} for obj in dash_objects]
                    live_data = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "faces": live_faces,
                        "objects": live_objs
                    }
                    with open("live_output.json", "w") as f:
                        json.dump(live_data, f, indent=4)
                except Exception as e:
                    logger.error(f"Failed to write live JSON: {e}")

                # Legacy speech queue processing is now handled by the interaction_strategy.on_speech callback
                pass

        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            logger.error(f"Error in agent loop: {e}", exc_info=True)
            self.stop()
