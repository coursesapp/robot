"""
Clean Face Tracking with YOLOv10 + ByteTrack
- Modern supervision API (version-safe)
- Proper letterbox resizing
- Correct frame alignment (0, 3, 6, 9...)
- No fragile tuple parsing
"""

import cv2
import time
import psutil
import torch
import numpy as np
import supervision as sv
from ultralytics import YOLO

# ============================================
# PATHS
# ============================================
MODEL_PATH = r"E:\Program Files\Downloads\yolov10n-face416.engine"
VIDEO_INPUT_PATH = r"E:\Program Files\Downloads\Walking While Using Phone (101).mp4"
OUTPUT_VIDEO_PATH = r"E:\Program Files\Downloads\YOLOv10n_FACE_CLEAN.mp4"

# ============================================
# CONFIG
# ============================================
DETECT_EVERY_N = 3
TRACK_EXPIRE_FRAMES = 30
MIN_CONF = 0.5
ENGINE_SIZE = 416

print("=" * 60)
print("Face Tracking - Clean Implementation")
print("=" * 60)

# ============================================
# LETTERBOX HELPER (proper aspect ratio preservation)
# ============================================
def letterbox_resize(img, new_shape=416):
    """Resize with letterbox padding to preserve aspect ratio"""
    shape = img.shape[:2]  # current shape [height, width]
    
    # Scale ratio (new / old)
    r = min(new_shape / shape[0], new_shape / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    return img, r, (dw, dh)

def scale_boxes(boxes, img_shape, ratio, pad):
    """Scale boxes from letterbox coordinates back to original image"""
    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= ratio
    
    # Clip to image boundaries
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_shape[0])
    
    return boxes

# ============================================
# TRACK STATE MANAGEMENT
# ============================================
class TrackManager:
    """Manages track persistence across frames"""
    
    def __init__(self, expire_frames=30):
        self.tracks = {}  # tid -> {'bbox', 'conf', 'last_seen'}
        self.expire_frames = expire_frames
    
    def update(self, detections: sv.Detections, frame_idx: int):
        """Update tracks from detections"""
        if detections.tracker_id is not None:
            for i in range(len(detections)):
                tid = int(detections.tracker_id[i])
                self.tracks[tid] = {
                    'bbox': detections.xyxy[i],
                    'conf': float(detections.confidence[i]) if detections.confidence is not None else 1.0,
                    'last_seen': frame_idx
                }
    
    def decay(self, frame_idx: int):
        """Remove expired tracks"""
        expired = [tid for tid, track in self.tracks.items() 
                   if frame_idx - track['last_seen'] > self.expire_frames]
        for tid in expired:
            del self.tracks[tid]
    
    def get_boxes(self):
        """Get all active tracks as list"""
        return [(tid, track['bbox'], track['conf']) 
                for tid, track in self.tracks.items()]

# ============================================
# LOAD MODEL & INITIALIZE
# ============================================
print("\nLoading model...")
model = YOLO(MODEL_PATH, task="detect")
print("✓ Model loaded")

print("Initializing ByteTrack...")
# Note: track_activation_threshold can be tuned to reduce ID jumping
bytetrack = sv.ByteTrack(track_activation_threshold=0.25)
track_manager = TrackManager(expire_frames=TRACK_EXPIRE_FRAMES)
print("✓ ByteTrack initialized")

# ============================================
# WARM-UP
# ============================================
if torch.cuda.is_available():
    print("\nWarming up TensorRT...")
    dummy = np.zeros((ENGINE_SIZE, ENGINE_SIZE, 3), dtype=np.uint8)
    for i in range(5):
        _ = model.predict(dummy, conf=MIN_CONF, verbose=False, device=0)
    print("✓ Warmup complete")

# ============================================
# VIDEO SETUP
# ============================================
print("\nOpening video...")
cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open video file")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 1:
    fps = 30.0  # Fallback

print(f"✓ Video: {frame_width}x{frame_height} @ {fps:.2f} FPS")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# ============================================
# METRICS
# ============================================
total_frames = 0
detection_count = 0
start_time = time.perf_counter()
track_times = []
draw_times = []

# ============================================
# MAIN LOOP
# ============================================
print("\nProcessing...")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx = total_frames
    
    # ====================
    # DETECTION PHASE
    # ====================
    t1 = time.perf_counter()
    
    if frame_idx % DETECT_EVERY_N == 0:
        # Letterbox resize (preserves aspect ratio)
        frame_resized, ratio, pad = letterbox_resize(frame, ENGINE_SIZE)
        
        # Run detection
        # Note: With TensorRT engine, input must already be ENGINE_SIZE
        # Do NOT pass imgsz parameter - it's ignored and misleading
        results = model.predict(
            frame_resized,
            conf=MIN_CONF,
            verbose=False,
            device=0 if torch.cuda.is_available() else 'cpu'
        )
        
        # Extract detections
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            
            # Scale boxes back to original image
            boxes = scale_boxes(boxes, (frame_height, frame_width), ratio, pad)
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            confs = np.empty((0,), dtype=np.float32)
            clss = np.empty((0,), dtype=np.int32)
        
        # Create supervision Detections
        detections = sv.Detections(
            xyxy=boxes,
            confidence=confs,
            class_id=clss
        )
        
        # Update ByteTrack
        tracked = bytetrack.update_with_detections(detections)
        
        # Update track manager
        track_manager.update(tracked, frame_idx)
        
        print(f"[Frame {frame_idx}] Detections: {len(detections)}, "
              f"Tracked: {len(tracked) if tracked.tracker_id is not None else 0}")
        
        detection_count += 1
    
    # Decay old tracks
    track_manager.decay(frame_idx)
    
    t2 = time.perf_counter()
    track_times.append((t2 - t1) * 1000)
    
    # ====================
    # DRAWING PHASE
    # ====================
    t3 = time.perf_counter()
    
    boxes_to_draw = track_manager.get_boxes()
    
    for tid, bbox, conf in boxes_to_draw:
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label with confidence
        label = f"ID:{tid} ({conf:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw stats
    cv2.putText(frame, f"Faces: {len(boxes_to_draw)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    t4 = time.perf_counter()
    draw_times.append((t4 - t3) * 1000)
    
    # Write frame
    out.write(frame)
    
    total_frames += 1
    
    # Progress
    if total_frames % 30 == 0:
        elapsed = time.perf_counter() - start_time
        current_fps = total_frames / elapsed
        print(f"Processed {total_frames} frames | FPS: {current_fps:.1f}")

# ============================================
# CLEANUP & METRICS
# ============================================
cap.release()
out.release()

elapsed = time.perf_counter() - start_time
avg_track = np.mean(track_times) if track_times else 0
avg_draw = np.mean(draw_times) if draw_times else 0
avg_fps = total_frames / elapsed

print("\n" + "=" * 60)
print("FINAL METRICS")
print("=" * 60)
print(f"Total Frames          : {total_frames}")
print(f"Detection Frames      : {detection_count} ({detection_count/total_frames*100:.1f}%)")
print(f"Total Time            : {elapsed:.2f}s")
print(f"Average FPS           : {avg_fps:.2f}")
print(f"Avg Tracking Step Time: {avg_track:.2f}ms")
print(f"Avg Drawing Time      : {avg_draw:.2f}ms")

if torch.cuda.is_available():
    vram = torch.cuda.memory_reserved() // 1024**2
    print(f"VRAM Used             : {vram}MB")

ram = psutil.Process().memory_info().rss // 1024**2
print(f"RAM Used              : {ram}MB")

print(f"\n✓ Output: {OUTPUT_VIDEO_PATH}")

print("=" * 60)
