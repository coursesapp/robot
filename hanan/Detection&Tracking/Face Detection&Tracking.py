import cv2
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

MODEL_PATH ="hanan\\Detection&Tracking\\resources\\yolov10n-face.pt"
OUTPUT_VIDEO_PATH ="hanan\\Detection&Tracking\\output\\Face_trackingg_output_camera.mp4"

class YoloDetector:
    def __init__(self, model_path, confidence):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, image):
        results = self.model.predict(image, conf=self.confidence, verbose=False)
        result = results[0]
        detections = self.make_detections(result, image)
        return detections

    def make_detections(self, result, frame):
        boxes = result.boxes
        detections = []

        if boxes is None:
            return detections

        frame_height, frame_width = frame.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            # Clip boxes to frame
            x1 = max(0, min(x1, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(0, min(x2, frame_width - 1))
            y2 = max(0, min(y2, frame_height - 1))

            w = x2 - x1
            h = y2 - y1

            class_number = int(box.cls[0])
            conf = float(box.conf[0])

            # DeepSORT expects (x, y, w, h)
            detections.append(((x1, y1, w, h), conf, class_number))

        return detections


class Tracker:
    def __init__(self):
        self.object_tracker = DeepSort(
            max_age=30,
            n_init=1,
            nms_max_overlap=0.7,
            max_cosine_distance=0.4,
            nn_budget=70,
            embedder="mobilenet",
            half=True,
            bgr=False   
        )

    def track(self, detections, frame):
        if not detections:
            return [], []

        tracks = self.object_tracker.update_tracks(detections, frame=frame)
        tracking_ids = []
        boxes = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            tracking_ids.append(track.track_id)
            left, top, right, bottom = map(int, track.to_ltrb())
            boxes.append((left, top, right, bottom))

        return tracking_ids, boxes


def main():
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.45)
    tracker = Tracker()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,18)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 10

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.perf_counter()
        detections = detector.detect(frame)
        tracking_ids, boxes = tracker.track(detections, frame)
        end_time = time.perf_counter()

        processing_time = end_time - start_time
        current_fps = 1.0 / processing_time if processing_time > 0 else 0

        for tracking_id, bounding_box in zip(tracking_ids, boxes):
            left, top, right, bottom = bounding_box

            if right > left and bottom > top:
                cv2.rectangle(frame, (left, top), (right, bottom), (100, 20, 0), 2)
                cv2.putText(frame, f"ID: {tracking_id}", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 20, 0), 2)

        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 20, 0), 2)
        cv2.putText(frame, f"Faces: {len(tracking_ids)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 20, 0), 2)

        out.write(frame)
        cv2.imshow("Live Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
