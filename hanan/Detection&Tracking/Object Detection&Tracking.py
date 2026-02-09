
import cv2
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

MODEL_PATH = "hanan\\Detection&Tracking\\resources\\yolov10n.pt"
OUTPUT_VIDEO_PATH = "hanan\\Detection&Tracking\\output\\Object_tracking_output_camera.mp4"

class YoloDetector:
    def __init__(self, model_path, confidence=0.2):
        self.model = YOLO(model_path)
        self.classList = ["person"]
        self.confidence = confidence

    def detect(self, image):
        results = self.model.predict(image, conf=self.confidence)
        result = results[0]
        detections = self.make_detections(result)
        return detections

    def make_detections(self, result):
        boxes = result.boxes
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            class_number = int(box.cls[0])
            
            if result.names[class_number] not in self.classList:
                continue

            conf = box.conf[0]
            detections.append((([x1, y1, w, h]), class_number, conf))
        return detections

class Tracker:
    def __init__(self):
        self.object_tracker = DeepSort(
            max_age=20,
            n_init=2,
            nms_max_overlap=0.3,
            max_cosine_distance=0.8,
            nn_budget=None,
            embedder="mobilenet",
            half=True,
            bgr=True
        )

    def track(self, detections, frame):
        tracks = self.object_tracker.update_tracks(detections, frame=frame)
        tracking_ids = []
        boxes = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            tracking_ids.append(track.track_id)
            ltrb = track.to_ltrb()
            boxes.append(ltrb)
        return tracking_ids, boxes

def main():
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.2)
    tracker = Tracker()

    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 

    print(f"Camera Info: {frame_width}x{frame_height}, {fps:.2f} FPS")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    total_processing_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        start_time = time.perf_counter()

        detections = detector.detect(frame)
        tracking_ids, boxes = tracker.track(detections, frame)

        end_time = time.perf_counter()

        processing_time = end_time - start_time
        total_processing_time += processing_time
        current_fps = 1 / processing_time if processing_time > 0 else 0

        for tracking_id, bounding_box in zip(tracking_ids, boxes):
            x1, y1, x2, y2 = map(int, bounding_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 20, 0), 2)
            cv2.putText(frame, f"ID: {tracking_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 20, 0), 2)

        cv2.putText(frame, f"FPS: {current_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 20, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 20, 0), 2)

        out.write(frame)
        cv2.imshow("Live Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

        frame_count += 1

    avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
    print(f"\n=== Processing Complete ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
