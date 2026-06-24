import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import math
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
MODEL_PATH = BASE_DIR / "yolo11n.pt"
EMERGENCY_KEYWORDS = ["ambulance", "fire", "police", "siren", "emergency", "rescue"]
VEHICLE_KEYWORDS = ["car", "truck", "bus", "motor", "bike", "vehicle", "van"]
PIXEL_TO_METER = 0.05

MODEL = None


def ensure_directories():
    UPLOAD_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)


def load_detector():
    global MODEL
    if MODEL is None:
        ensure_directories()
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")
        MODEL = YOLO(str(MODEL_PATH))
    return MODEL


def safe_filename(filename: str) -> str:
    return filename.replace(" ", "_").replace("/", "_").replace("\\", "_")


def uploaded_path(filename: str) -> Path:
    return UPLOAD_DIR / safe_filename(filename)


def processed_video_path(filename: str) -> Path:
    safe_name = safe_filename(filename)
    name = Path(safe_name).stem
    return PROCESSED_DIR / f"{name}_processed.mp4"


def heatmap_path(filename: str) -> Path:
    safe_name = safe_filename(filename)
    name = Path(safe_name).stem
    return PROCESSED_DIR / f"{name}_heatmap.png"


def video_exists(filename: str) -> bool:
    return uploaded_path(filename).exists()


def _get_label_names():
    model = load_detector()
    return getattr(model, "names", {}) or {}


def _is_vehicle_label(label: str) -> bool:
    return any(keyword in label.lower() for keyword in VEHICLE_KEYWORDS)


def _is_emergency_label(label: str) -> bool:
    return any(keyword in label.lower() for keyword in EMERGENCY_KEYWORDS)


def _compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height
    if inter_area == 0:
        return 0.0
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / float(boxA_area + boxB_area - inter_area)


class CentroidTracker:
    def __init__(self, max_distance=60):
        self.next_id = 0
        self.objects = {}
        self.max_distance = max_distance

    def update(self, centroids):
        if len(self.objects) == 0:
            for centroid in centroids:
                self.objects[self.next_id] = {"centroid": centroid, "previous": None, "counted": False}
                self.next_id += 1
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[obj_id]["centroid"] for obj_id in object_ids]
        distance_matrix = np.linalg.norm(np.array(object_centroids)[:, None] - np.array(centroids)[None, :], axis=2)

        assigned_columns = set()
        updated = {}

        for row_index, obj_id in enumerate(object_ids):
            distances = distance_matrix[row_index]
            col_index = int(np.argmin(distances))
            if col_index in assigned_columns:
                updated[obj_id] = self.objects[obj_id]
                continue
            if distances[col_index] > self.max_distance:
                updated[obj_id] = self.objects[obj_id]
                continue
            updated[obj_id] = {
                "centroid": tuple(centroids[col_index]),
                "previous": self.objects[obj_id]["centroid"],
                "counted": self.objects[obj_id]["counted"],
            }
            assigned_columns.add(col_index)

        for index, centroid in enumerate(centroids):
            if index not in assigned_columns:
                updated[self.next_id] = {"centroid": tuple(centroid), "previous": None, "counted": False}
                self.next_id += 1

        self.objects = updated
        return self.objects


def get_video_info(filename: str) -> str:
    path = uploaded_path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Uploaded video not found: {filename}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("Unable to open the uploaded video.")

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frames / fps if fps else 0
    cap.release()

    return (
        f"Video '{filename}': {frames} frames, {fps:.1f} FPS, {width}x{height} px, "
        f"duration {duration:.1f} seconds."
    )


def _parse_boxes(result):
    labels = _get_label_names()
    detections = []
    for box in result.boxes:
        xyxy = box.xyxy[0].tolist()
        category_id = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
        label = labels.get(category_id, f"class_{category_id}")
        detections.append(
            {
                "box": [int(coord) for coord in xyxy],
                "confidence": float(box.conf[0]) if hasattr(box.conf[0], "item") else float(box.conf[0]),
                "class_id": category_id,
                "label": label,
            }
        )
    return detections


def _detect_lanes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=80, maxLineGap=120)
    lane_image = image.copy()
    line_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 200, 100), 3)
            line_count += 1
    return lane_image, line_count


def _overlay_heatmap(image, points):
    height, width = image.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    for x, y in points:
        if 0 <= y < height and 0 <= x < width:
            heatmap[int(y), int(x)] += 1
    if np.max(heatmap) == 0:
        return image, None
    heatmap = cv2.GaussianBlur(heatmap, (47, 47), 0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
    return overlay, heatmap_color


def _estimate_speed(previous, current, fps):
    if previous is None or current is None or fps <= 0:
        return 0.0
    dx = current[0] - previous[0]
    dy = current[1] - previous[1]
    distance_pixels = math.sqrt(dx * dx + dy * dy)
    meters = distance_pixels * PIXEL_TO_METER
    seconds = 1.0 / fps
    speed_mps = meters / seconds
    return speed_mps * 3.6


def analyze_video_sample(filename: str, frame_step: int = 5) -> dict:
    path = uploaded_path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Uploaded video not found: {filename}")

    model = load_detector()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    vehicle_count = 0
    emergency_count = 0
    avg_speed = []
    collisions = []
    heat_points = []
    lane_count_max = 0
    label_counts = {}

    tracker = CentroidTracker(max_distance=70)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame % frame_step != 0:
            current_frame += 1
            continue

        results = model(frame)
        if len(results) == 0:
            current_frame += 1
            continue
        detections = _parse_boxes(results[0])
        centers = []

        for det in detections:
            label_counts[det["label"]] = label_counts.get(det["label"], 0) + 1
            if _is_vehicle_label(det["label"]):
                x1, y1, x2, y2 = det["box"]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                centers.append((cx, cy))
                heat_points.append((cx, cy))
                vehicle_count += 1
                if _is_emergency_label(det["label"]):
                    emergency_count += 1

        objects = tracker.update(centers)
        for obj_id, info in objects.items():
            speed = _estimate_speed(info["previous"], info["centroid"], fps)
            if speed > 0:
                avg_speed.append(speed)

        lane_frame, lane_count = _detect_lanes(frame)
        lane_count_max = max(lane_count_max, lane_count)

        # simple accident detection based on nearby vehicle overlap
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                if _is_vehicle_label(detections[i]["label"]) and _is_vehicle_label(detections[j]["label"]):
                    if _compute_iou(detections[i]["box"], detections[j]["box"]) > 0.25:
                        collisions.append((detections[i]["label"], detections[j]["label"]))

        current_frame += 1
        if current_frame > frame_count or len(avg_speed) >= 50:
            break

    cap.release()
    return {
        "sampled_frames": min(frame_count // frame_step + 1, current_frame // frame_step + 1),
        "vehicle_count": vehicle_count,
        "average_speed_kmh": round(sum(avg_speed) / len(avg_speed), 1) if avg_speed else 0.0,
        "emergency_vehicle_count": emergency_count,
        "accident_alerts": len(collisions),
        "collision_examples": collisions[:5],
        "heat_points": heat_points,
        "lane_count": lane_count_max,
        "label_counts": label_counts,
        "tracked_vehicles": len(tracker.objects),
    }


def predict_congestion_stats(vehicle_count: int, average_speed_kmh: float) -> str:
    score = min(4, max(0, round((vehicle_count / 20) + (1.0 - min(average_speed_kmh, 100) / 100) * 2)))
    levels = ["Free flow", "Light", "Moderate", "Heavy", "Severe"]
    return f"Predicted congestion level: {levels[score]} (score={score})"


def detect_lanes_summary(filename: str) -> str:
    path = uploaded_path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Uploaded video not found: {filename}")
    cap = cv2.VideoCapture(str(path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Unable to read a frame for lane detection.")
    _, lane_count = _detect_lanes(frame)
    return f"Detected {lane_count} lane line(s) in the initial frame."


def classification_summary(filename: str) -> str:
    summary = analyze_video_sample(filename)
    if not summary["label_counts"]:
        return "No classified objects found in the sampled frames."
    counts = ", ".join(f"{label}: {count}" for label, count in summary["label_counts"].items())
    return f"Vehicle classification counts: {counts}"


def vehicle_detection_summary(filename: str) -> str:
    summary = analyze_video_sample(filename)
    if not summary["label_counts"]:
        return "No objects detected in the sampled frames."
    vehicle_counts = {
        label: count
        for label, count in summary["label_counts"].items()
        if _is_vehicle_label(label)
    }
    if not vehicle_counts:
        return "No vehicles detected in the sampled frames."
    counts = ", ".join(f"{label}: {count}" for label, count in sorted(vehicle_counts.items()))
    return f"Vehicle detection summary: {counts}"


def vehicle_count_summary(filename: str) -> str:
    summary = analyze_video_sample(filename)
    return (
        f"Sampled vehicle count: {summary['vehicle_count']}. "
        f"Unique tracked objects: {summary['tracked_vehicles']}."
    )


def speed_estimate_summary(filename: str) -> str:
    summary = analyze_video_sample(filename)
    if summary["average_speed_kmh"] <= 0:
        return "Unable to estimate vehicle speed from sampled frames."
    return (
        f"Estimated average speed: {summary['average_speed_kmh']} km/h across sampled frames. "
        f"This is based on centroid movement from {summary['sampled_frames']} sampled frames."
    )


def vehicle_tracking_summary(filename: str) -> str:
    summary = analyze_video_sample(filename)
    return f"Tracked unique vehicles: {summary['tracked_vehicles']}"


def lane_detection_summary(filename: str) -> str:
    summary = analyze_video_sample(filename)
    lane_count = summary.get("lane_count", summary.get("lane_count_max", 0))
    return f"Detected lane structures in the sampled frames: {lane_count} lines observed."


def process_full_video(filename: str) -> dict:
    path = uploaded_path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Uploaded video not found: {filename}")

    ensure_directories()
    model = load_detector()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = processed_video_path(filename)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker = CentroidTracker(max_distance=60)
    all_heat_points = []
    collision_alerts = []
    emergency_detected = set()
    total_speed = []
    total_vehicles = 0
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = _parse_boxes(results[0]) if len(results) else []
        centers = []
        line_y = int(height * 0.7)
        vehicle_status = []

        for det in detections:
            label = det["label"]
            x1, y1, x2, y2 = det["box"]
            if _is_vehicle_label(label):
                total_vehicles += 1
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                centers.append((cx, cy))
                all_heat_points.append((cx, cy))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 160, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
                vehicle_status.append(label)
                if _is_emergency_label(label):
                    emergency_detected.add(label)

        objects = tracker.update(centers)
        avg_speed_frame = []
        for obj_id, info in objects.items():
            speed = _estimate_speed(info["previous"], info["centroid"], fps)
            if speed > 0:
                avg_speed_frame.append(speed)
            if info["previous"] is not None:
                cv2.circle(frame, (int(info["centroid"][0]), int(info["centroid"][1])), 4, (0, 255, 0), -1)
                cv2.putText(frame, f"ID {obj_id}", (int(info["centroid"][0]) + 6, int(info["centroid"][1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)

        if avg_speed_frame:
            total_speed.extend(avg_speed_frame)

        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                if _is_vehicle_label(detections[i]["label"]) and _is_vehicle_label(detections[j]["label"]):
                    if _compute_iou(detections[i]["box"], detections[j]["box"]) > 0.35:
                        collision_alerts.append((detections[i]["label"], detections[j]["label"], frame_index))
                        cv2.putText(frame, "POSSIBLE ACCIDENT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        cv2.rectangle(frame, (20, 20), (520, 80), (0, 0, 255), 2)

        lane_img, lane_count = _detect_lanes(frame)
        if lane_count > 0:
            frame = lane_img
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 90), 2)

        info_text = f"Vehicles: {total_vehicles} | Avg speed: {round(sum(avg_speed_frame) / len(avg_speed_frame), 1) if avg_speed_frame else 0:.1f} km/h"
        cv2.putText(frame, info_text, (18, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    heatmap_image, _ = _overlay_heatmap(np.zeros((height, width, 3), dtype=np.uint8), all_heat_points)
    heatmap_file = heatmap_path(filename)
    cv2.imwrite(str(heatmap_file), heatmap_image)

    average_speed = round(sum(total_speed) / len(total_speed), 1) if total_speed else 0.0
    prediction = predict_congestion_stats(total_vehicles, average_speed)
    return {
        "processed_video": output_path,
        "heatmap_image": heatmap_file,
        "total_vehicles": total_vehicles,
        "avg_speed_kmh": average_speed,
        "collision_count": len(collision_alerts),
        "emergency_labels": sorted(list(emergency_detected)),
        "prediction": prediction,
    }


def quick_summary(filename: str) -> str:
    summary = analyze_video_sample(filename)
    congestion = predict_congestion_stats(summary["vehicle_count"], summary["average_speed_kmh"])
    result_parts = [
        f"Sampled frames: {summary['sampled_frames']}",
        f"Vehicle detections: {summary['vehicle_count']}",
        f"Avg estimated speed: {summary['average_speed_kmh']} km/h",
        f"Emergency vehicles: {summary['emergency_vehicle_count']}",
        f"Accident confidence alerts: {summary['accident_alerts']}",
        congestion,
    ]
    if summary["collision_examples"]:
        result_parts.append(f"Collision sample labels: {summary['collision_examples']}")
    return "\n".join(result_parts)


def generate_congestion_heatmap(filename: str) -> Path:
    summary = analyze_video_sample(filename)
    path = heatmap_path(filename)
    cap = cv2.VideoCapture(str(uploaded_path(filename)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    heatmap_image, _ = _overlay_heatmap(np.zeros((height, width, 3), dtype=np.uint8), summary["heat_points"])
    cv2.imwrite(str(path), heatmap_image)
    return path


def detect_emergency_vehicles(filename: str) -> str:
    summary = analyze_video_sample(filename)
    if summary["emergency_vehicle_count"] > 0:
        return f"Emergency vehicles detected: {summary['emergency_vehicle_count']} in the sampled frames."
    return "No emergency vehicles detected in the sampled frames."


def detect_accidents(filename: str) -> str:
    summary = analyze_video_sample(filename)
    if summary["accident_alerts"] > 0:
        return f"Possible accident alerts found: {summary['accident_alerts']} times."
    return "No accident alerts detected in the sampled frames."


def predict_congestion(filename: str) -> str:
    summary = analyze_video_sample(filename)
    return predict_congestion_stats(summary["vehicle_count"], summary["average_speed_kmh"])


def live_stream_generator():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam for live stream.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
