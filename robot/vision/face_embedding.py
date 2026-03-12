import numpy as np
import cv2
import logging
from typing import List, Optional, Tuple, Dict
from pathlib import Path

logger = logging.getLogger("FaceEmbedder")
# now we use simple libraties instead of InceptionResnetV1 , MTCNN , facenet-pytorch / PyTorch
class FaceEmbedder:
    def __init__(self, 
                 det_model_path: str = "models/face_detection_yunet_2023mar.onnx",
                 rec_model_path: str = "models/face_recognition_sface_2021dec.onnx",
                 ctx_id: int = 0, 
                 det_size: Tuple[int, int] = (640, 640)):
        
        self.det_size = det_size
        
        # Check model existence
        if not Path(det_model_path).exists() or not Path(rec_model_path).exists():
            logger.error(f"Face models not found at {det_model_path} / {rec_model_path}")
            self.detector = None
            self.recognizer = None
            return

        # Initialize YuNet (Face Detection)
        self.detector = cv2.FaceDetectorYN.create(
            det_model_path,
            "",
            det_size,
            0.6, # Score threshold
            0.3, # NMS threshold
            5000 # Top K
        )
        
        # Initialize SFace (Face Recognition)
        self.recognizer = cv2.FaceRecognizerSF.create(
            rec_model_path,
            ""
        )
        logger.info("OpenCV Face models loaded.")

    def extract(self, img: np.ndarray) -> Optional[Dict]:
        """
        Returns the largest face embedding and info.
        """
        if self.detector is None or self.recognizer is None:
            return None

        h, w, _ = img.shape
        self.detector.setInputSize((w, h))

        # Detect
        # faces -> [x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
        _, faces = self.detector.detect(img)
        
        if faces is None or len(faces) == 0:
            return None
            
        # Get largest face (w * h)
        # faces is shape (1, 15) or (N, 15)
        if len(faces.shape) > 1 and faces.shape[0] > 1:
            areas = faces[:, 2] * faces[:, 3]
            max_idx = np.argmax(areas)
            face = faces[max_idx]
        else:
            # Single face case where faces might be (1, 15)
            # Or depending on OpenCV version, just a 1D array
            face = faces[0] if len(faces.shape) > 1 else faces

        # Align and Extract
        # SFace expects aligned face crop
        aligned_face = self.recognizer.alignCrop(img, face)
        embedding = self.recognizer.feature(aligned_face)
        
        # Normalize embedding (SFace output is usually normalized but good to ensure for cosine sim)
        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
            
        # Convert bbox to x1,y1,x2,y2
        x1, y1, w_box, h_box = face[0:4]
        bbox = np.array([x1, y1, x1+w_box, y1+h_box]).astype(int)

        return {
            'embedding': embedding, # 128-d usually for SFace
            'bbox': bbox,
            'kps': None, # YuNet landmarks are different format, skipping for now
        }
