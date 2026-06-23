from pathlib import Path
import shutil
import app
import video_features

BASE = Path(__file__).resolve().parent
UPLOAD = BASE / 'uploads'
UPLOAD.mkdir(exist_ok=True)

source_video = BASE / 'traffic.mp4'
if source_video.exists():
    dest = UPLOAD / 'traffic.mp4'
    shutil.copy(source_video, dest)
    print('Copied traffic.mp4 to uploads')
else:
    print('traffic.mp4 not found in workspace root')

for action in [
    'video_info',
    'quick_summary',
    'congestion_prediction',
    'detect_emergency',
    'detect_accident',
    'lane_detection',
    'classification',
    'tracking',
]:
    try:
        print('ACTION', action)
        result = app.run_feature(action, 'traffic.mp4')
        print(result)
    except Exception as e:
        print('ERROR', action, type(e).__name__, e)

try:
    print('FULL_PROCESS start')
    res = video_features.process_full_video('traffic.mp4')
    print('FULL_PROCESS', res)
except Exception as e:
    print('FULL_PROCESS ERROR', type(e).__name__, e)
