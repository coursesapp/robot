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

    def extract(self, img: np.ndarray, box: Optional[Tuple[int, int, int, int]] = None, extract_embedding: bool = True) -> Optional[Dict]:
        """
        Returns the largest face embedding and info.
        Optional 'box' (x1, y1, x2, y2) from YOLO/Detector can be provided to focus detection.
        If extract_embedding is False, it only calculates landmarks/mouth_dist (fast).
        """
        if self.detector is None:
            return None
        if extract_embedding and self.recognizer is None:
            return None

        # Use box to crop if provided (improves performance and accuracy)
        offset_x, offset_y = 0, 0
        if box is not None:
            bx1, by1, bx2, by2 = box
            ih, iw, _ = img.shape
            # Add margin
            margin = 20
            bx1 = max(0, bx1 - margin)
            by1 = max(0, by1 - margin)
            bx2 = min(iw, bx2 + margin)
            by2 = min(ih, by2 + margin)
            
            img = img[by1:by2, bx1:bx2]
            if img.size == 0: return None
            offset_x, offset_y = bx1, by1

        h, w, _ = img.shape
        self.detector.setInputSize((w, h))

        # Detect
        # faces -> [x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
        _, faces = self.detector.detect(img)
        
        if faces is None or len(faces) == 0:
            return None
            
        # Get largest face (w * h)
        if len(faces.shape) > 1 and faces.shape[0] > 1:
            areas = faces[:, 2] * faces[:, 3]
            max_idx = np.argmax(areas)
            face = faces[max_idx]
        else:
            face = faces[0] if len(faces.shape) > 1 else faces

        # Align and Extract
        embedding = None
        if extract_embedding:
            aligned_face = self.recognizer.alignCrop(img, face)
            embedding = self.recognizer.feature(aligned_face)
            
            # Normalize embedding (Crucial for FAISS Inner Product/Cosine similarity)
            embedding = embedding.flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm
            
        # Convert bbox to x1,y1,x2,y2 and add offsets
        fx1, fy1, fw, fh = face[0:4]
        bbox = np.array([fx1 + offset_x, fy1 + offset_y, fx1 + offset_x + fw, fy1 + offset_y + fh]).astype(int)

        # Calculate Pose (Frontal Score)
        # Landmarks: [right_eye, left_eye, nose, right_mouth_corner, left_mouth_corner]
        re_x, re_y = face[4:6]
        le_x, le_y = face[6:8]
        nt_x, nt_y = face[8:10]
        
        # Calculate horizontal ratios to check if nose is centered
        dist_re_le = max(1, abs(re_x - le_x))
        dist_re_nt = abs(re_x - nt_x)
        dist_le_nt = abs(le_x - nt_x)
        
        # frontal_score: 1.0 = perfectly centered, 0.0 = extreme side profile
        asymmetry = abs(dist_re_nt - dist_le_nt) / dist_re_le
        frontal_score = max(0.0, 1.0 - asymmetry)

        # Calculate Mouth Logic for Speaker Detection
        # Dist from nose to mouth center
        mouth_cx = (face[10] + face[12]) / 2
        mouth_cy = (face[11] + face[13]) / 2
        nose_to_mouth = np.sqrt((nt_x - mouth_cx)**2 + (nt_y - mouth_cy)**2)

        return {
            'embedding': embedding,
            'bbox': bbox,
            'frontal_score': float(frontal_score),
            'landmarks': face[4:14].tolist(),
            'mouth_dist': float(nose_to_mouth / max(1, fw)) # Normalized by face width
        }
