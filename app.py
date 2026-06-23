import os
import sys
import subprocess
from pathlib import Path
from flask import (
    Flask,
    request,
    redirect,
    url_for,
    render_template,
    send_from_directory,
    flash,
    jsonify,
    has_app_context,
)
import analytics_csv
import video_features

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(BASE_DIR / UPLOAD_FOLDER)
app.config["PROCESSED_FOLDER"] = str(BASE_DIR / PROCESSED_FOLDER)
app.secret_key = "replace-this-with-a-secure-key"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["PROCESSED_FOLDER"], exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_uploaded_files():
    uploads = []
    upload_dir = Path(app.config["UPLOAD_FOLDER"])
    if not upload_dir.exists():
        return uploads

    for path in sorted(upload_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.is_file():
            uploads.append(
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "modified": path.stat().st_mtime,
                    "type": path.suffix.lstrip(".").lower(),
                }
            )
    return uploads


def get_file_info(filename):
    path = Path(app.config["UPLOAD_FOLDER"]) / filename
    if not path.exists() or not path.is_file():
        return None

    return {
        "name": filename,
        "path": str(path),
        "size": path.stat().st_size,
        "type": path.suffix.lstrip(".").lower(),
    }


def processed_file_link(filename: str) -> str:
    if has_app_context():
        return url_for("processed_file", filename=filename)
    with app.test_request_context():
        return url_for("processed_file", filename=filename)


def run_feature(action, filename):
    if action == "video_info":
        info = get_file_info(filename)
        if info is None:
            raise FileNotFoundError(f"Uploaded file not found: {filename}")
        return (
            f"Uploaded file {info['name']} is stored at {info['path']} "
            f"({info['size']} bytes, .{info['type']})."
        )

    if action == "generate_analytics":
        analytics_csv.main()
        return "Traffic analytics generated successfully. Check traffic_analytics.csv."

    if action == "train_prediction":
        script_path = BASE_DIR / "traffic_prediction.py"
        completed = subprocess.run(
            [sys.executable, str(script_path), "--data", "traffic_analytics.csv"],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip())
        return completed.stdout.strip()

    if action == "quick_summary":
        return video_features.quick_summary(filename)

    if action == "congestion_prediction":
        return video_features.predict_congestion(filename)

    if action == "generate_heatmap":
        heatmap_path = video_features.generate_congestion_heatmap(filename)
        link = processed_file_link(heatmap_path.name)
        return f"Heatmap generated successfully. <a href=\"{link}\" target=\"_blank\">View heatmap</a>."

    if action == "vehicle_detection":
        return video_features.vehicle_detection_summary(filename)

    if action == "vehicle_count":
        return video_features.vehicle_count_summary(filename)

    if action == "speed_estimation":
        return video_features.speed_estimate_summary(filename)

    if action == "detect_emergency":
        return video_features.detect_emergency_vehicles(filename)

    if action == "detect_accident":
        return video_features.detect_accidents(filename)

    if action == "lane_detection":
        return video_features.detect_lanes_summary(filename)

    if action == "classification":
        return video_features.classification_summary(filename)

    if action in {"tracking", "vehicle_tracking"}:
        return video_features.vehicle_tracking_summary(filename)

    if action == "full_process":
        result = video_features.process_full_video(filename)
        video_link = processed_file_link(Path(result["processed_video"]).name)
        heatmap_link = processed_file_link(Path(result["heatmap_image"]).name)
        lines = [
            f"Processed video is ready: <a href=\"{video_link}\" target=\"_blank\">Open video</a>",
            f"Heatmap is available: <a href=\"{heatmap_link}\" target=\"_blank\">View heatmap</a>",
            f"Total vehicles: {result['total_vehicles']}",
            f"Average speed: {result['avg_speed_kmh']} km/h",
            f"Collisions flagged: {result['collision_count']}",
            f"Emergency vehicles detected: {', '.join(result['emergency_labels']) if result['emergency_labels'] else 'None'}",
            f"Prediction: {result['prediction']}",
        ]
        return "<br>".join(lines)

    raise ValueError(f"Unsupported feature action: {action}")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "video" not in request.files:
            flash("No video file part.")
            return redirect(request.url)

        file = request.files["video"]
        if file.filename == "":
            flash("No selected file.")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = file.filename
            save_path = Path(app.config["UPLOAD_FOLDER"]) / filename
            file.save(save_path)
            return redirect(url_for("upload_success", filename=filename))

        flash("Invalid file type. Allowed types: mp4, avi, mov, mkv.")
        return redirect(request.url)

    return render_template("index.html", uploads=get_uploaded_files())


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/processed/<filename>")
def processed_file(filename):
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)


@app.route("/live")
def live():
    return render_template("live.html")


@app.route("/video_feed")
def video_feed():
    return app.response_class(
        video_features.live_stream_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/success/<filename>")
def upload_success(filename):
    file_url = url_for("uploaded_file", filename=filename)
    file_info = get_file_info(filename)
    return render_template(
        "success.html",
        filename=filename,
        file_url=file_url,
        file_info=file_info,
        uploads=get_uploaded_files(),
    )


@app.route("/process", methods=["POST"])
def process_action():
    data = request.get_json(force=True)
    filename = data.get("filename")
    action = data.get("action")

    if not filename or not action:
        return jsonify(status="error", message="Missing filename or action."), 400

    try:
        result = run_feature(action, filename)
        return jsonify(status="success", message=result)
    except Exception as exc:
        return jsonify(status="error", message=str(exc)), 400


if __name__ == "__main__":
    app.run(debug=True)
