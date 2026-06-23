import os
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "traffic.mp4"
MODEL_PATH = "yolo11n.pt"
LINE_Y = 300


class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_id = 0
        self.objects = {}
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_id] = {
            "centroid": centroid,
            "previous_centroid": None,
            "counted": False,
        }
        self.next_id += 1

    def update(self, centroids):
        if len(self.objects) == 0:
            for centroid in centroids:
                self.register(centroid)
            return self.objects

        object_ids = list(self.objects.keys())
        previous_centroids = [self.objects[obj_id]["centroid"] for obj_id in object_ids]
        if len(previous_centroids) == 0 or len(centroids) == 0:
            return self.objects

        distance_matrix = np.linalg.norm(
            np.array(previous_centroids)[:, None] - np.array(centroids)[None, :], axis=2
        )
        assigned_cols = set()
        updated_objects = {}

        for row_index, obj_id in enumerate(object_ids):
            col_index = np.argmin(distance_matrix[row_index])
            if col_index in assigned_cols:
                updated_objects[obj_id] = self.objects[obj_id]
                continue

            distance = distance_matrix[row_index, col_index]
            if distance > self.max_distance:
                updated_objects[obj_id] = self.objects[obj_id]
                continue

            updated_objects[obj_id] = {
                "centroid": tuple(centroids[col_index]),
                "previous_centroid": self.objects[obj_id]["centroid"],
                "counted": self.objects[obj_id]["counted"],
            }
            assigned_cols.add(col_index)

        for col_index, centroid in enumerate(centroids):
            if col_index not in assigned_cols:
                self.register(tuple(centroid))

        for obj_id in object_ids:
            if obj_id not in updated_objects:
                updated_objects[obj_id] = self.objects[obj_id]

        self.objects = updated_objects
        return self.objects


def main():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"YOLO model file not found: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {VIDEO_PATH}")

    tracker = CentroidTracker(max_distance=50)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        tracked_centroids = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                tracked_centroids.append((cx, cy))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        objects = tracker.update(tracked_centroids)
        for obj_id, data in objects.items():
            prev = data["previous_centroid"]
            current = data["centroid"]
            if prev is None or data["counted"]:
                continue

            if prev[1] < LINE_Y <= current[1] or prev[1] > LINE_Y >= current[1]:
                count += 1
                data["counted"] = True

        frame_width = frame.shape[1]
        cv2.line(frame, (0, LINE_Y), (frame_width, LINE_Y), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Count: {count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Vehicle Counting", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()