import cv2
from pathlib import Path

# -------- CONFIG --------
VIDEO_PATH = Path("data/videos/zed_index0_sbs_2560x720.mp4")  # change per video
OUTPUT_ROOT = Path("data/frames")  # everything goes under here
# ------------------------

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

# Use the video filename (without extension) as the folder name
video_name = VIDEO_PATH.stem  # e.g. "zed_index0_sbs_2560x720"
out_dir = OUTPUT_ROOT / video_name
out_left = out_dir / "left"
out_right = out_dir / "right"

out_left.mkdir(parents=True, exist_ok=True)
out_right.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    mid = w // 2

    left = frame[:, :mid]
    right = frame[:, mid:]

    cv2.imwrite(str(out_left / f"left_{frame_id:06d}.png"), left)
    cv2.imwrite(str(out_right / f"right_{frame_id:06d}.png"), right)

    frame_id += 1

cap.release()

print(f"Done. Extracted {frame_id} frames from {VIDEO_PATH.name}")
print(f"Saved to: {out_dir.resolve()}")


# python preprocessing/extract_and_split_sbs.py
