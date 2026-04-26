import cv2
import json
from datetime import datetime
from pathlib import Path

VIDEOS_DIR = Path("data/videos")
OUTPUT_ROOT = Path("data/frames")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

video_files = sorted(list(VIDEOS_DIR.glob("*.mp4")) + list(VIDEOS_DIR.glob("*.mov")))
if not video_files:
    raise RuntimeError(f"No videos found in {VIDEOS_DIR.resolve()}")

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

for video_path in video_files:
    video_name = video_path.stem
    out_dir = OUTPUT_ROOT / video_name
    out_left = out_dir / "left"
    out_right = out_dir / "right"
    out_left.mkdir(parents=True, exist_ok=True)
    out_right.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[SKIP] Could not open {video_path.name}")
        continue

    # Read video properties (best effort)
    fps = safe_float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    extracted = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        mid = w // 2

        left = frame[:, :mid]
        right = frame[:, mid:]

        cv2.imwrite(str(out_left / f"left_{extracted:06d}.png"), left)
        cv2.imwrite(str(out_right / f"right_{extracted:06d}.png"), right)

        extracted += 1

    cap.release()

    # Metadata (useful for reproducibility + dissertation methods)
    metadata = {
        "video_file": video_path.name,
        "video_stem": video_name,
        "processed_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "opencv_version": cv2.__version__,
        "input_video_properties": {
            "fps_reported": fps,
            "frame_count_reported": frame_count,
            "width_reported": width,
            "height_reported": height
        },
        "extraction": {
            "frames_extracted": extracted,
            "split_mode": "side_by_side",
            "left_frame_shape": [height, width // 2, 3] if width else None,
            "right_frame_shape": [height, width // 2, 3] if width else None,
            "output_left_dir": str(out_left),
            "output_right_dir": str(out_right),
            "filename_pattern_left": "left_%06d.png",
            "filename_pattern_right": "right_%06d.png"
        }
    }

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] {video_path.name}: extracted {extracted} frames -> {out_dir}")
    print(f"     metadata -> {meta_path}")

print("Batch complete.")



# to run
# source .venv/bin/activate   
# python preprocessing/batch_extract_and_split.py













# import cv2
# from pathlib import Path

# VIDEOS_DIR = Path("data/videos")
# OUTPUT_ROOT = Path("data/frames")

# OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# video_files = sorted(list(VIDEOS_DIR.glob("*.mp4")) + list(VIDEOS_DIR.glob("*.mov")))
# if not video_files:
#     raise RuntimeError(f"No videos found in {VIDEOS_DIR.resolve()}")

# for video_path in video_files:
#     video_name = video_path.stem
#     out_dir = OUTPUT_ROOT / video_name
#     out_left = out_dir / "left"
#     out_right = out_dir / "right"
#     out_left.mkdir(parents=True, exist_ok=True)
#     out_right.mkdir(parents=True, exist_ok=True)

#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         print(f"[SKIP] Could not open {video_path.name}")
#         continue

#     frame_id = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         h, w, _ = frame.shape
#         mid = w // 2
#         left = frame[:, :mid]
#         right = frame[:, mid:]

#         cv2.imwrite(str(out_left / f"left_{frame_id:06d}.png"), left)
#         cv2.imwrite(str(out_right / f"right_{frame_id:06d}.png"), right)
#         frame_id += 1

#     cap.release()
#     print(f"[OK] {video_path.name}: {frame_id} frames -> {out_dir}")

# print("Batch complete.")
