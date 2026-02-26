"""
realtime_fer.py
---------------
Real-time webcam emotion detection using the FER library with MTCNN face detection.
FER is chosen here for its low latency on CPU, making it suitable for live video feeds.

Usage:
    python realtime_fer.py

Controls:
    q  —  quit
"""

import threading
import cv2
from fer import FER

# ── Configuration ────────────────────────────────────────────────────────────
CAMERA_INDEX = 0        # 0 = default webcam; change for external cameras
FRAME_SKIP = 5          # Analyze every Nth frame (higher = faster display, less frequent updates)
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
# ─────────────────────────────────────────────────────────────────────────────

# Shared state between display thread and analysis thread
latest_label = ""
analysis_lock = threading.Lock()


def analyze_frame(detector: FER, frame) -> None:
    """Run FER emotion analysis in a background thread and update latest_label."""
    global latest_label
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    emotion, score = detector.top_emotion(rgb_frame)
    if emotion:
        with analysis_lock:
            latest_label = f"{emotion} ({score:.2f})"


def main() -> None:
    detector = FER(mtcnn=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

    print("Emotion detection started. Press 'q' to quit.")

    frame_count = 0
    active_thread = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Lost camera feed.")
            break

        frame_count += 1

        # Launch analysis in background thread every FRAME_SKIP frames
        if frame_count % FRAME_SKIP == 0:
            if active_thread is None or not active_thread.is_alive():
                active_thread = threading.Thread(
                    target=analyze_frame,
                    args=(detector, frame.copy()),
                    daemon=True
                )
                active_thread.start()

        # Always overlay the most recent result (non-blocking)
        with analysis_lock:
            label = latest_label

        if label:
            cv2.putText(
                frame, label, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 230, 0), 2, cv2.LINE_AA
            )

        cv2.imshow("Live Emotion Detection (FER) — press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
