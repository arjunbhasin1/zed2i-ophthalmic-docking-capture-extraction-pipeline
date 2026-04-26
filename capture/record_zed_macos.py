import cv2
import time
from pathlib import Path
import re

# -------- CONFIG --------
CAM_INDEX = 0
OUT_DIR = Path("data/videos")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_W, TARGET_H = 2560, 720
TARGET_FPS = 30
SHOW_PREVIEW = True
# ------------------------

# Find next available incremental filename: vid_000.mp4, vid_001.mp4, ...
pattern = re.compile(r"^vid_(\d{3})\.mp4$")

existing_ids = []
for p in OUT_DIR.glob("vid_*.mp4"):
    m = pattern.match(p.name)
    if m:
        existing_ids.append(int(m.group(1)))

next_id = (max(existing_ids) + 1) if existing_ids else 0
OUT_PATH = OUT_DIR / f"vid_{next_id:03d}.mp4"

print(f"[INFO] Saving new recording to: {OUT_PATH.resolve()}")

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

# Request settings (macOS may not honor all requests, so we read back actual values)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"[INFO] Opened camera {CAM_INDEX}: {actual_w}x{actual_h} @ {actual_fps:.2f} fps")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(OUT_PATH), fourcc, TARGET_FPS, (actual_w, actual_h))

print("[INFO] Recording... press 'q' to stop.")
start = time.time()
frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Failed to read frame.")
        break

    out.write(frame)
    frames += 1

    if SHOW_PREVIEW:
        cv2.imshow("ZED index 0 (SBS)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

elapsed = time.time() - start
print(f"[INFO] Saved {frames} frames in {elapsed:.1f}s (~{frames/elapsed:.1f} fps)")
print(f"[INFO] Video saved to: {OUT_PATH.resolve()}")

cap.release()
out.release()
cv2.destroyAllWindows()


#to run:
# source .venv/bin/activate
# python capture/record_zed_macos.py
