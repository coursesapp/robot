import cv2
import logging
import numpy as np
import threading
from typing import Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("EmotionClassifier")

# now work on cpu only 

class EmotionClassifier:
    def __init__(self, model_path: Optional[str] = None):
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']
        self.session = None
        self.input_name = None
        self.output_name = None
        self.lock = threading.Lock()
        
        if model_path and Path(model_path).exists():
            import onnxruntime as ort
            try:
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                logger.info(f"Loaded emotion model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load emotion model: {e}")
        else:
            logger.warning("No emotion model provided or found. Using dummy predictions.")

    def predict(self, face_img: np.ndarray) -> Dict[str, float]:
        """
        Takes a BGR face crop, returns probabilities for each emotion.
        """
        if self.session is None:
            return {'neutral': 0.9, 'happy': 0.1}

        try:
            # Preprocess
            img = cv2.resize(face_img, (64, 64)) # Assuming standard ONNX input size
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=0) # (1, 1, 64, 64)
            
            # Predict
            with self.lock:
                outputs = self.session.run([self.output_name], {self.input_name: img})
                scores = outputs[0][0] # logits
                
            # Softmax
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
            
            result = {}
            for i, label in enumerate(self.labels):
                if i < len(probs):
                    result[label] = float(probs[i])
            return result

        except Exception as e:
            logger.error(f"Emotion inference error: {e}")
            return {'neutral': 1.0}
