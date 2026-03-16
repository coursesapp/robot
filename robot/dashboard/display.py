import cv2
import numpy as np
import threading
import time
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class DashboardState:
    # Camera
    frame: np.ndarray = None
    tracks: List[Any] = field(default_factory=list)
    known_people: Dict[int, str] = field(default_factory=dict)
    
    # Status
    camera_fps: int = 0
    stt_status: str = "Idle"
    llm_status: str = "Idle"
    
    # Logs
    event_log: List[str] = field(default_factory=list) # Full history
    
    def add_event(self, event: str):
        timestamp = time.strftime("%H:%M:%S")
        self.event_log.append(f"[{timestamp}] {event}")

class Dashboard:
    def __init__(self, window_name="Agent Dashboard", width=1280, height=720):
        self.window_name = window_name
        self.width = width
        self.height = height
        self.state = DashboardState()
        self.lock = threading.Lock()
        self.running = False
        
        # Display config
        self.sidebar_width = 400
        self.log_height = 250
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        self.thread = threading.Thread(target=self._render_loop, daemon=True)

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def update_state(self, update_fn):
        with self.lock:
            update_fn(self.state)

    def _render_loop(self):
        while self.running:
            with self.lock:
                # Copy state for rendering
                frame = self.state.frame.copy() if self.state.frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                tracks = list(self.state.tracks)
                known_people = dict(self.state.known_people)
                stt_status = self.state.stt_status
                llm_status = self.state.llm_status
                fps = self.state.camera_fps
                events = list(self.state.event_log)

            # --- Layout Dimensions ---
            # Create main canvas
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 30) # Dark gray background
            
            # Sub-areas
            cam_x, cam_y = 20, 20
            cam_w = self.width - self.sidebar_width - 60
            cam_h = self.height - self.log_height - 60
            
            sidebar_x = cam_x + cam_w + 20
            sidebar_y = 20
            sidebar_h = self.height - 40
            
            log_x = 20
            log_y = cam_y + cam_h + 20
            log_w = cam_w
            
            # --- Draw Camera Feed ---
            if frame is not None:
                # Resize keeping aspect ratio
                fh, fw = frame.shape[:2]
                scale = min(cam_w/fw, cam_h/fh)
                new_w, new_h = int(fw*scale), int(fh*scale)
                resized = cv2.resize(frame, (new_w, new_h))
                
                # Draw boxes/labels on resized frame
                for track in tracks:
                    if track.class_name == 'person':
                        x1, y1, x2, y2 = track.box
                        # Scale coords to resized frame
                        sx1, sy1 = int(x1*scale), int(y1*scale)
                        sx2, sy2 = int(x2*scale), int(y2*scale)
                        
                        pid = known_people.get(track.track_id, "unknown")
                        color = (0, 255, 0) if pid != "unknown" else (0, 0, 255)
                        
                        cv2.rectangle(resized, (sx1, sy1), (sx2, sy2), color, 2)
                        
                        # Label
                        label = f"ID:{track.track_id} {pid}"
                        (tw, th), _ = cv2.getTextSize(label, self.font, 0.5, 1)
                        cv2.rectangle(resized, (sx1, sy1-th-5), (sx1+tw, sy1), color, -1)
                        cv2.putText(resized, label, (sx1, sy1-5), self.font, 0.5, (0,0,0), 1)

                # Center the camera frame in its area
                offset_x = cam_x + (cam_w - new_w)//2
                offset_y = cam_y + (cam_h - new_h)//2
                canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
                cv2.rectangle(canvas, (offset_x, offset_y), (offset_x+new_w, offset_y+new_h), (100,100,100), 2)

            # --- Draw Sidebar (Status & Identities) ---
            cv2.rectangle(canvas, (sidebar_x, sidebar_y), (sidebar_x+self.sidebar_width, sidebar_y+sidebar_h), (40,40,40), -1)
            cv2.rectangle(canvas, (sidebar_x, sidebar_y), (sidebar_x+self.sidebar_width, sidebar_y+sidebar_h), (100,100,100), 1)
            
            # Status Section
            text_y = sidebar_y + 30
            cv2.putText(canvas, "SYSTEM STATUS", (sidebar_x+20, text_y), self.font, 0.7, (255,255,255), 2)
            cv2.line(canvas, (sidebar_x+20, text_y+10), (sidebar_x+self.sidebar_width-20, text_y+10), (100,100,100), 1)
            
            text_y += 40
            cv2.putText(canvas, f"Camera FPS: {fps}", (sidebar_x+20, text_y), self.font, 0.6, (200,200,200), 1)
            text_y += 30
            stt_color = (0, 255, 0) if "listen" in stt_status.lower() else (200, 200, 200)
            cv2.putText(canvas, f"Mic: {stt_status}", (sidebar_x+20, text_y), self.font, 0.6, stt_color, 1)
            text_y += 30
            llm_color = (0, 255, 255) if "thinking" in llm_status.lower() else (200, 200, 200)
            cv2.putText(canvas, f"LLM: {llm_status}", (sidebar_x+20, text_y), self.font, 0.6, llm_color, 1)

            # People Section
            text_y += 50
            cv2.putText(canvas, "PEOPLE IN FRAME", (sidebar_x+20, text_y), self.font, 0.7, (255,255,255), 2)
            cv2.line(canvas, (sidebar_x+20, text_y+10), (sidebar_x+self.sidebar_width-20, text_y+10), (100,100,100), 1)
            
            text_y += 40
            if not known_people:
                cv2.putText(canvas, "No people detected.", (sidebar_x+20, text_y), self.font, 0.6, (150,150,150), 1)
            
            for tid, pid in known_people.items():
                status = "KNOWN" if pid != "unknown" else "NEW"
                color = (0, 255, 0) if pid != "unknown" else (0, 0, 255)
                cv2.putText(canvas, f"[{status}] {pid} (TID:{tid})", (sidebar_x+20, text_y), self.font, 0.6, color, 1)
                text_y += 30

            # --- Draw Event Log ---
            cv2.rectangle(canvas, (log_x, log_y), (log_x+log_w, log_y+self.log_height), (40,40,40), -1)
            cv2.rectangle(canvas, (log_x, log_y), (log_x+log_w, log_y+self.log_height), (100,100,100), 1)
            
            cv2.putText(canvas, "EVENT LOG", (log_x+10, log_y+25), self.font, 0.6, (255,255,255), 1)
            cv2.line(canvas, (log_x+10, log_y+35), (log_x+log_w-10, log_y+35), (100,100,100), 1)
            
            # Show last N lines fitting in the box
            line_height = 25
            max_lines = (self.log_height - 50) // line_height
            visible_events = events[-max_lines:]
            
            ev_y = log_y + 60
            for ev in visible_events:
                # Basic wrap (approx)
                if len(ev) > 120:
                    ev = ev[:117] + "..."
                
                # Color code
                col = (200, 200, 200)
                if "Agent:" in ev: col = (255, 200, 0)
                elif "User:" in ev or "Heard:" in ev: col = (0, 200, 255)
                elif "Embedding" in ev or "identity" in ev: col = (0, 255, 0)
                elif "Thought:" in ev or "🧠" in ev: col = (255, 100, 255) # Pink for reasoning
                elif "Action:" in ev or "⚙️" in ev: col = (100, 255, 255) # Yellow-Cyan for actions
                
                cv2.putText(canvas, ev, (log_x+10, ev_y), self.font, 0.5, col, 1)
                ev_y += line_height

            # Show window
            cv2.imshow(self.window_name, canvas)
            
            key = cv2.waitKey(30)
            if key == 27: # ESC
                self.running = False
                break
                
        cv2.destroyWindow(self.window_name)
