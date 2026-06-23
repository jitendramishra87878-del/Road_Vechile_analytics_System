import argparse
import os
import cv2
from ultralytics import YOLO

MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = "traffic.mp4"


def process_video(video_path=VIDEO_PATH, model_path=MODEL_PATH):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model file not found: {model_path}")

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        cv2.imshow("Vehicle Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Run YOLO vehicle detection on a video.")
    parser.add_argument(
        "--video",
        default=VIDEO_PATH,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="Path to the YOLO model file.",
    )
    args = parser.parse_args()

    process_video(video_path=args.video, model_path=args.model)


if __name__ == "__main__":
    main()