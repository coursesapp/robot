from ultralytics import YOLO
import logging
from typing import List, Tuple
from dataclasses import dataclass

logger = logging.getLogger("Detector")

@dataclass
class TrackedObject:
    track_id: int
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_name: str
    confidence: float

class Detector:
    def __init__(self, 
                 object_model_path: str = "models/yolov8n.pt", 
                 face_model_path: str = "models/yolov8n-face.pt", 
                 conf_threshold: float = 0.45):
        logger.info(f"Loading YOLO object model from {object_model_path}...")
        try:
            self.object_model = YOLO(object_model_path)
            self.face_model = YOLO(face_model_path)
            self.conf_threshold = conf_threshold
            logger.info("YOLO models loaded.")
        except Exception as e:
            logger.error(f"Failed to load YOLO models: {e}")
            raise e

    def detect_and_track(self, frame) -> List[TrackedObject]:
        if frame is None:
            return []
            
        tracks = []
        
        # 1. Object Tracking
        # leave GPU for embedding models and LLM 
        object_results = self.object_model.track(
            frame,
            conf=self.conf_threshold,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            device='cpu'
        )[0]
        
        if object_results.boxes.id is not None:
            for box, track_id in zip(object_results.boxes, object_results.boxes.id):
                cls_id = int(box.cls[0])
                class_name = self.object_model.names[cls_id]
                
                # Filter out 'person' from general object model since we use face model for people
                if class_name == 'person':
                    continue
                    
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                
                tracks.append(TrackedObject(
                    track_id=int(track_id),
                    box=(int(x1), int(y1), int(x2), int(y2)),
                    class_name=class_name,
                    confidence=conf
                ))
                
        # 2. Face Tracking (maps to 'person' in our agent logic)
        face_results = self.face_model.track(
            frame,
            conf=self.conf_threshold,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            device='cpu'
        )[0]
        
        if face_results.boxes.id is not None:
            for box, track_id in zip(face_results.boxes, face_results.boxes.id):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                
                # Offset track_id so they don't collide with object track_ids
                face_tid = int(track_id) + 100000 
                
                tracks.append(TrackedObject(
                    track_id=face_tid,
                    box=(int(x1), int(y1), int(x2), int(y2)),
                    class_name='person',
                    confidence=conf
                ))

        return tracks
