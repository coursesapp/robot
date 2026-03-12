import json
import queue
import threading
import time
import logging
import cv2
from dataclasses import dataclass, field
from typing import Any, Dict, List
from flask import Flask, Response, render_template, jsonify

logger = logging.getLogger("WebDashboard")


@dataclass
class DashboardState:
    # Status
    camera_fps: int = 0
    stt_status: str = "Idle"
    llm_status: str = "Idle"
    llm_busy: bool = False
    
    # People
    known_people: Dict[int, str] = field(default_factory=dict)
    
    # Detected objects with spatial awareness
    objects: List[Dict] = field(default_factory=list)
    
    # Camera frame
    frame: Any = None
    
    # Dedicated thought tracking
    current_thought: str = ""
    start_time: float = field(default_factory=time.time)
    
    # Event log (full history for /log) and pending events for SSE
    event_log: List[str] = field(default_factory=list)
    new_events: List[str] = field(default_factory=list)
    
    def add_event(self, event: str):
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {event}"
        self.event_log.append(log_entry)
        self.new_events.append(log_entry)
        
        # Memory leak prevention (cap at 300)
        if len(self.event_log) > 300:
            self.event_log.pop(0)


class WebDashboard:
    def __init__(self, host="0.0.0.0", port=5050):
        self.host = host
        self.port = port
        self.state = DashboardState()
        self.lock = threading.Lock()
        
        # SSE client queue pool: each connected client gets its own queue
        self._sse_queues: List[queue.Queue] = []
        self._sse_queues_lock = threading.Lock()

        self.app = Flask(__name__, template_folder="templates")
        self._register_routes()
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)

    def start(self):
        self._server_thread.start()
        logger.info(f"Web Dashboard started at http://localhost:{self.port}")

    def stop(self):
        pass  # daemon thread exits with main process

    def update_state(self, update_fn):
        """Thread-safe state update. Broadcasts change to all SSE clients."""
        with self.lock:
            update_fn(self.state)
        self._broadcast_state()

    def _broadcast_state(self):
        """Push current state to all connected SSE clients."""
        with self.lock:
            snapshot = {
                "camera_fps": self.state.camera_fps,
                "stt_status": self.state.stt_status,
                "llm_status": self.state.llm_status,
                "llm_busy": self.state.llm_busy,
                "known_people": {str(k): v for k, v in self.state.known_people.items()},
                "objects": self.state.objects,
                "uptime": int(time.time() - self.state.start_time),
                "current_thought": self.state.current_thought,
                "new_events": list(self.state.new_events),
            }
            # Clear pending events after copying them to snapshot
            self.state.new_events.clear()
        
        payload = f"data: {json.dumps(snapshot)}\n\n"
        with self._sse_queues_lock:
            dead = []
            for q in self._sse_queues:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._sse_queues.remove(q)

    def _register_routes(self):
        app = self.app

        @app.route("/")
        def index():
            return render_template("index.html", port=self.port)
            
        @app.route("/video_feed")
        def video_feed():
            def generate_frames():
                while True:
                    frame = getattr(self.state, 'frame', None)
                    if frame is not None:
                        ret, buffer = cv2.imencode('.jpg', frame)
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    time.sleep(0.1)  # ~10 FPS preview stream
            return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route("/events")
        def events():
            client_q = queue.Queue(maxsize=100)
            with self._sse_queues_lock:
                self._sse_queues.append(client_q)

            def generate():
                try:
                    while True:
                        try:
                            # 15s timeout to send a heartbeat and keep connection alive
                            payload = client_q.get(timeout=15)
                            yield payload
                        except queue.Empty:
                            # Heartbeat to prevent browser from dropping SSE
                            yield ": heartbeat\n\n"
                except GeneratorExit:
                    pass
                finally:
                    with self._sse_queues_lock:
                        if client_q in self._sse_queues:
                            self._sse_queues.remove(client_q)

            return Response(generate(), mimetype="text/event-stream",
                            headers={"Cache-Control": "no-cache",
                                     "X-Accel-Buffering": "no"})

        @app.route("/log")
        def full_log():
            with self.lock:
                return jsonify(self.state.event_log)

    def _run_server(self):
        import logging as _logging
        _logging.getLogger("werkzeug").setLevel(_logging.ERROR)
        self.app.run(host=self.host, port=self.port, threaded=True, use_reloader=False)
