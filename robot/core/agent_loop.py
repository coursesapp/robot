import time
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
from dialogue.llm_client import LLMClient
from dialogue.prompt_engine import PromptEngine
from dialogue.info_extractor import InfoExtractor
from dialogue.summarizer import Summarizer
from memory.identity_store import IdentityStore
from memory.social_memory import SocialMemory
from memory.vector_memory import VectorMemory
from dashboard.web_dashboard import WebDashboard
from core.actions import ActionLibrary
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
        # Context state
        self.current_speech = queue.Queue()
        self.known_people = {} # track_id -> person_id
        self.vote_history = {} # track_id -> deque(maxlen=5)
        self.context_history = [] # List[str]
        self.seen_object_ids = set() # set of track_id

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
        self.stt = STTEngine(cfg['audio']['whisper_model'], callback=self.on_speech)
        self.stt.start()
        
        self.tts = TTSEngine(cfg['audio']['piper_model'], cfg['audio'].get('piper_config'))
        self.tts.start()

        # Dialogue (Supports Ollama or llama-server via config)
        self.llm = LLMClient(cfg.get('llm', {}))
        self.prompt_engine = PromptEngine()
        self.info_extractor = InfoExtractor(self.llm)
        self.summarizer = Summarizer(self.llm)
        self.action_library = ActionLibrary()
        
        # State for proactive behavior
        self.last_greeted = {} # track_id -> timestamp
        self.turn_count = 0
        
        # Dashboard
        self.dashboard.start()

        logger.info("All modules initialized.")

    def on_speech(self, text: str):
        self.current_speech.put(text)
        self.context_history.append(f"User: {text}")
        
        # Update Dashboard Log
        def update_stt(state):
            state.add_event(f"Heard: {text}")
            state.stt_status = "Processing..." # Reverts to listening shortly maybe? Let's just keep it Listening usually.
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
                
                self.last_frame_time = time.time()
                self.frame_count += 1
                
                # --- 1. PERCEIVE ---
                frame = self.camera.get_frame()
                if frame is None:
                    continue

                tracks = self.detector.detect_and_track(frame)
                
                # Identify People
                current_identities = []
                current_emotions = []
                
                for track in tracks:
                    if track.class_name == 'person':
                        # Simple approach: If track doesn't have ID, run face rec
                        pid = self.known_people.get(track.track_id)
                        
                        if not pid or self.frame_count % 5 == 0: # Re-verify every 5 frames (~2.5s at 2 FPS)
                            # Crop to track box with margin
                            h, w, _ = frame.shape
                            x1, y1, x2, y2 = track.box
                            face_res = self.face_embedder.extract(frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)])
                            
                            if face_res:
                                pid, is_new = self.identity_store.find_or_create(face_res['embedding'], threshold=self.config['face']['similarity_threshold'])
                                
                                # Log to dashboard if this track didn't have this PID before
                                if self.known_people.get(track.track_id) != pid:
                                    if is_new:
                                        def up_new(st, p=pid): st.add_event(f"📌 New identity: {p}")
                                        self.dashboard.update_state(up_new)
                                    else:
                                        soc_name = self.social_memory.get(pid).get('name')
                                        disp_name = soc_name if soc_name else pid
                                        def up_known(st, d=disp_name): st.add_event(f"✅ Known: {d}")
                                        self.dashboard.update_state(up_known)
                                        
                                self.known_people[track.track_id] = pid
                                
                                # Proactive greeting logic moved below after identity settling
                                            
                                # Emotion
                                if self.frame_count % self.config['vision']['emotion_every_n_frames'] == 0:
                                    emo_probs = self.emotion_classifier.predict(frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)])
                                    top_emo = max(emo_probs, key=emo_probs.get)
                                    current_emotions.append(top_emo)
                            else:
                                pid = "unknown"
                        
                        # Apply Identity Voting
                        if track.track_id not in self.vote_history:
                            self.vote_history[track.track_id] = deque(maxlen=5)
                        
                        # Only append if we got a fresh reading or it's unknown
                        if not self.known_people.get(track.track_id) or self.frame_count % 5 == 0 or pid == "unknown":
                            self.vote_history[track.track_id].append(pid)
                        
                        if self.vote_history[track.track_id]:
                            final_pid = max(set(self.vote_history[track.track_id]), key=self.vote_history[track.track_id].count)
                        else:
                            final_pid = pid
                            
                        # Log if changed
                        if self.known_people.get(track.track_id) != final_pid:
                            self.known_people[track.track_id] = final_pid
                            if final_pid != "unknown":
                                soc_name = self.social_memory.get(final_pid).get('name')
                                disp = soc_name if soc_name else final_pid
                                self.dashboard.update_state(lambda st, d=disp: st.add_event(f"✅ Identity Settled: {d}"))

                        if final_pid and final_pid != "unknown":
                            current_identities.append(final_pid)
                            
                            # Proactive Greeting Verification (Debounced & using SETTLED identity)
                            now_t = time.time()
                            last_g_time = self.last_greeted.get(final_pid, 0)
                            cooldown = self.config.get('agent', {}).get('greet_cooldown', 120)
                            
                            # Require at least 5 frames of tracking stability before speaking
                            if len(self.vote_history[track.track_id]) >= 5 and (now_t - last_g_time > cooldown):
                                has_name = bool(self.social_memory.get(final_pid).get('name'))
                                if not has_name:
                                    self.current_speech.put(f"[System Event: You just saw a person (ID: {final_pid}) whose name you don't know yet. Greet them warmly and ask for their name.]")
                                    self.last_greeted[final_pid] = now_t
                                else:
                                    sociality = self.config.get('agent', {}).get('sociality', 0.7)
                                    if random.random() < sociality:
                                        self.current_speech.put(f"[System Event: You just saw a known person (ID: {final_pid}). Start a friendly conversation referring to their past context or facts.]")
                                        self.last_greeted[final_pid] = now_t
                    else:
                        # Non-person Object Snapshot
                        if track.track_id not in self.seen_object_ids:
                            self.seen_object_ids.add(track.track_id)
                            h, w, _ = frame.shape
                            x1, y1, x2, y2 = track.box
                            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                            if crop.size > 0:
                                os.makedirs("memory/object_snapshots", exist_ok=True)
                                cv2.imwrite(f"memory/object_snapshots/{track.class_name}_{track.track_id}.jpg", crop)

                # --- Update Dashboard Vision State ---
                display_names = {}
                for tid, p in self.known_people.items():
                    if p != "unknown":
                        mem = self.social_memory.get(p)
                        n = mem.get('name')
                        name_str = n if n else p
                        
                        display_names[tid] = {
                            "name": name_str,
                            "emotion": "",
                            "job": mem.get('job', ""),
                            "interests": mem.get('interests', []),
                            "is_known_pid": True
                        }
                    else:
                        display_names[tid] = {
                            "name": "unknown", 
                            "emotion": "", 
                            "is_known_pid": False
                        }
                        
                # Add emotion to the last detected person if available
                # current_emotions stores the emotions detected in this frame
                for i, tid in enumerate(display_names.keys()):
                    if i < len(current_emotions):
                        display_names[tid]["emotion"] = current_emotions[i]

                # Format objects for dashboard
                dash_objects = []
                h, w, _ = frame.shape
                for t in tracks:
                    if t.class_name != 'person':
                        cx = (t.box[0] + t.box[2]) / 2.0
                        pos = "center"
                        if cx < w * 0.33: pos = "left"
                        elif cx > w * 0.66: pos = "right"
                        dash_objects.append({"label": t.class_name, "position": pos})

                def update_vision(state, dn=display_names, objs=dash_objects, fps=self.fps, fr=frame):
                    state.camera_fps = fps
                    state.known_people = dn
                    state.objects = objs
                    state.frame = fr
                self.dashboard.update_state(update_vision)

                # --- Live JSON Output ---
                try:
                    live_faces = [{"id": tid, "name": name} for tid, name in display_names.items() if name != "unknown"]
                    live_objs = [{"label": t.class_name, "position": f"{t.box}"} for t in tracks if t.class_name != 'person']
                    live_data = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "faces": live_faces,
                        "objects": live_objs
                    }
                    with open("live_output.json", "w") as f:
                        json.dump(live_data, f, indent=4)
                except Exception as e:
                    logger.error(f"Failed to write live JSON: {e}")


                # --- 2. UNDERSTAND & DECIDE ---
                # Check for speech
                try:
                    user_text = self.current_speech.get_nowait()
                except queue.Empty:
                    user_text = None

                if user_text and not self.llm_busy:
                    # Snapshot environment state now (don't wait for LLM)
                    soc_data = [self.social_memory.get(pid) for pid in current_identities]
                    
                    # Spatial Objects Parsing
                    objects_spatial = []
                    nh, nw, _ = frame.shape
                    for t in tracks:
                        if t.class_name != 'person':
                            x1, y1, x2, y2 = t.box
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                            pos_x = "left" if cx < nw/3 else "right" if cx > 2*nw/3 else "center"
                            pos_y = "top" if cy < nh/3 else "bottom" if cy > 2*nh/3 else "middle"
                            objects_spatial.append({"label": t.class_name, "position": f"{pos_y}-{pos_x}"})
                    
                    def update_obj(st, objs=objects_spatial):
                        st.objects = objs
                    self.dashboard.update_state(update_obj)
                    
                    deep_memory = []
                    if current_identities and not user_text.startswith("[System"):
                        pid = current_identities[0]
                        deep_memory = self.vector_memory.search_past(pid, user_text, n_results=3)

                    context = {
                        "identities": current_identities,
                        "social_data": soc_data,
                        "emotions": current_emotions,
                        "objects": objects_spatial,
                        "history": list(self.context_history),
                        "deep_memory": deep_memory,
                        "available_actions": self.action_library.get_available_actions_schema(),
                        "time": time.ctime()
                    }                    
                    prompt = self.prompt_engine.build_prompt(user_text, context)
                    
                    # Dispatch LLM call in background (no blocking!)
                    self.llm_busy = True
                    def _llm_worker(prompt=prompt, utext=user_text, identities=list(current_identities)):
                        try:
                            # LLM Generation
                            logger.info("Thinking...")
                            self.dashboard.update_state(lambda st: setattr(st, 'llm_status', 'Thinking...') or setattr(st, 'llm_busy', True))
                            
                            response = self.llm.generate(prompt, max_tokens=self.cfg.get('llm', {}).get('max_tokens', 1024))
                            self.dashboard.update_state(lambda st: setattr(st, 'llm_status', 'Idle') or setattr(st, 'llm_busy', False))
                            
                            # --- JSON Parsing & Action Execution ---
                            response_audio = response  # Fallback
                            try:
                                json_str = response
                                if "```json" in json_str:
                                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                                elif "```" in json_str:
                                    json_str = json_str.split("```")[1].split("```")[0].strip()
                                    
                                data = json.loads(json_str)
                                
                                thought = data.get("internal_thought", "")
                                if thought:
                                    logger.info(f"🧠 Thought: {thought}")
                                    
                                    def update_thought(st, t=thought):
                                        st.current_thought = t
                                        st.add_event(f"🧠 Thought: {t}")
                                        
                                    self.dashboard.update_state(update_thought)
                                    
                                response_audio = data.get("response", "") or response
                                
                                actions = data.get("actions", [])
                                for act in actions:
                                    act_name = act.get("action")
                                    act_params = act.get("parameters", {})
                                    if act_name:
                                        self.action_library.execute(act_name, act_params)
                                        self.dashboard.update_state(lambda st, a=act_name: st.add_event(f"⚙️ Action: {a}"))

                                # --- Handle save_to_memory from new prompt schema ---
                                save_to_memory = data.get("save_to_memory", False)
                                summary_text = data.get("summary", "")
                                if save_to_memory and summary_text and identities:
                                    pid0 = identities[0]
                                    self.social_memory.update(pid0, {"summary": summary_text})
                                    self.vector_memory.add_interaction(pid0, f"[Key Info]: {summary_text}", role="system")
                                    self.dashboard.update_state(lambda st, s=summary_text: st.add_event(f"💾 Saved to memory: {s[:60]}..."))
                                        
                            except json.JSONDecodeError:
                                logger.warning(f"Non-JSON LLM response, using raw text.")
                                self.dashboard.update_state(lambda st: st.add_event(f"⚠️ Non-JSON response."))
                                
                            self.dashboard.update_state(lambda st, r=response_audio: st.add_event(f"Agent: {r}"))
                            logger.info(f"Agent: {response_audio}")
                            
                            self.context_history.append(f"user: {utext}")
                            self.context_history.append(f"agent: {response_audio}")
                            
                            # --- Smart context compression (from RAG notebook pattern) ---
                            # When history reaches 10 exchanges, summarize and compress to 1 entry
                            if len(self.context_history) >= 10:
                                try:
                                    summary = self.summarizer.summarize(list(self.context_history))
                                    if summary:
                                        self.context_history = [f"[Conversation Summary]: {summary}"]
                                        if identities:
                                            pid0 = identities[0]
                                            self.social_memory.update(pid0, {"summary": summary})
                                            self.vector_memory.add_interaction(pid0, f"[Conversation Summary]: {summary}", role="system")
                                        self.dashboard.update_state(lambda st: st.add_event(f"📝 Context compressed (summarized)"))
                                except Exception as se:
                                    logger.warning(f"Summarizer failed: {se}")
                            
                            # --- 3. ACT ---
                            self.tts.speak(response_audio)

                            # --- 4. MEMORY PIPELINE ---
                            self.turn_count += 1
                            
                            # Ingest to VectorMemory
                            if identities:
                                pid0 = identities[0]
                                if not utext.startswith("[System"):
                                    self.vector_memory.add_interaction(pid0, utext, role="user")
                                self.vector_memory.add_interaction(pid0, response_audio, role="agent")
                            
                            # Run InfoExtractor
                            if identities and not utext.startswith("[System"):
                                facts = self.info_extractor.extract(utext)
                                if facts:
                                    self.social_memory.update(identities[0], facts)
                                    self.dashboard.update_state(lambda st, fk=list(facts.keys()): st.add_event(f"💾 Facts saved: {fk}"))
                                    
                        except Exception as e:
                            logger.error(f"LLM Worker error: {e}")
                        finally:
                            self.llm_busy = False

                    threading.Thread(target=_llm_worker, daemon=True).start()

        except KeyboardInterrupt:
            self.stop()

        except Exception as e:
            logger.error(f"Error in agent loop: {e}", exc_info=True)
            self.stop()
