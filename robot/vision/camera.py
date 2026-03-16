import cv2
import threading
import time
import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger("Camera")

class Camera:
    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.stopped = False
        self.frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.cap = None

        # Sync initialisation to catch failures immediately vs threaded wait
        logger.info(f"Opening camera {device_id}...")
        self.cap = cv2.VideoCapture(self.device_id)
        
        # Give some time for sensors to warm up
        time.sleep(1.0)
        
        if not self.cap.isOpened():
            logger.error(f"Cannot open camera {self.device_id}")
            self.stopped = True
            return
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Start capture thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        # Wait for first frame
        logger.info(f"Opening camera {device_id}...")
        start = time.time()
        while self.frame is None and time.time() - start < 5.0:
            time.sleep(0.1)
        
        if self.frame is None:
            logger.error("Failed to get first frame from camera!")
        else:
            logger.info("Camera started successfully.")

    def _capture_loop(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to grab frame. Reconnecting...")
                self.cap.release()
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.device_id)
                continue

            with self.lock:
                self.frame = frame
            
            # Simple throttle to prevent CPU burn if camera is very fast
            # changed from 0.01 to 0.1
            time.sleep(0.1)

        self.cap.release()

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        return None

    def stop(self):
        self.stopped = True
        self.thread.join()
