import cv2
import json
import time
import re
import numpy as np
from pathlib import Path
import sys

# ---------------- CONFIG ----------------
CAM_INDEX = 0
TARGET_W, TARGET_H = 2560, 720
CAMERA_FPS = 30
CAPTURE_FPS = 3
SHOW_PREVIEW = True

VIDEOS_DIR = Path("data/videos")
FRAMES_DIR = Path("data/frames")

# -------- ZOOM SETTINGS --------
zoom_factor = 1.0
zoom_step = 0.2
max_zoom = 5.0
min_zoom = 1.0

# -------- TROCAR TRACKING --------
auto_track = False
last_center = None
smooth_alpha = 0.7
# ---------------------------------------

def print_controls():
    print("\n[CONTROLS]")
    print("   +   Zoom In")
    print("   -   Zoom Out")
    print("   t   Toggle Trocar Tracking")
    print("   q   Quit\n")

def print_zoom():
    sys.stdout.write(f"\r[ZOOM] {zoom_factor:.1f}x     ")
    sys.stdout.flush()

def print_tracking_status():
    status = "enabled" if auto_track else "disabled"
    sys.stdout.write(f"\r[TROCAR TRACKING] {status}     ")
    sys.stdout.flush()

def apply_digital_zoom(frame, zoom, center=None):
    if zoom == 1.0:
        return frame
    h, w, _ = frame.shape
    new_w = int(w / zoom)
    new_h = int(h / zoom)
    cx, cy = (w // 2, h // 2) if center is None else center
    x1 = max(0, cx - new_w // 2)
    y1 = max(0, cy - new_h // 2)
    x2 = min(w, x1 + new_w)
    y2 = min(h, y1 + new_h)
    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h))

def detect_trocar_center(frame):
    h, w, _ = frame.shape
    roi = frame[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=50,
        param1=120, param2=30,
        minRadius=5, maxRadius=120
    )
    if circles is None:
        return None
    x, y, r = np.uint16(np.around(circles))[0][0]
    return (int(x + w*0.2), int(y + h*0.2)), r

def next_video_path(videos_dir: Path) -> Path:
    videos_dir.mkdir(parents=True, exist_ok=True)
    ids = [
        int(m.group(1)) for p in videos_dir.glob("vid_*.mp4")
        if (m := re.match(r"vid_(\d{3})\.mp4", p.name))
    ]
    next_id = max(ids) + 1 if ids else 0
    return videos_dir / f"vid_{next_id:03d}.mp4"

def record_video(out_path: Path) -> bool:
    global zoom_factor, auto_track, last_center

    print_controls()
    print_zoom()

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    out = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        CAPTURE_FPS,
        (TARGET_W, TARGET_H)
    )

    frame_interval = 1 / CAPTURE_FPS
    last_capture_time = 0
    quit_requested = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ----- Swap left/right for live preview and recording -----
        mid = frame.shape[1] // 2
        frame[:, :mid], frame[:, mid:] = frame[:, mid:].copy(), frame[:, :mid].copy()

        zoom_center = None
        if auto_track:
            result = detect_trocar_center(frame)
            if result:
                (cx, cy), r = result
                last_center = (cx, cy) if last_center is None else (
                    int(smooth_alpha*last_center[0] + (1-smooth_alpha)*cx),
                    int(smooth_alpha*last_center[1] + (1-smooth_alpha)*cy)
                )
                zoom_center = last_center
                zoom_factor = 3.5
                cv2.circle(frame, last_center, r, (0, 255, 0), 2)

        frame_zoomed = apply_digital_zoom(frame, zoom_factor, zoom_center)
        now = time.time()
        if now - last_capture_time >= frame_interval:
            out.write(frame_zoomed)
            last_capture_time = now

        if SHOW_PREVIEW:
            cv2.imshow("ZED Trocar Capture", frame_zoomed)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            quit_requested = True
            break
        elif key == ord("t"):
            auto_track = not auto_track
            last_center = None
            print_tracking_status()
            print()  # move to next line for clarity
        elif key in (ord("+"), ord("=")):
            zoom_factor = min(max_zoom, zoom_factor + zoom_step)
            print_zoom()
        elif key == ord("-"):
            zoom_factor = max(min_zoom, zoom_factor - zoom_step)
            print_zoom()

    cap.release()
    out.release()

    if quit_requested:
        confirm = np.zeros((300, 600, 3), dtype=np.uint8)
        cv2.putText(confirm, "Keep this recording?",
                    (90, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(confirm, "Y = Yes    N = No",
                    (140, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        while True:
            cv2.imshow("Confirm", confirm)
            k = cv2.waitKey(0) & 0xFF
            if k == ord("y"):
                cv2.destroyAllWindows()
                return True
            elif k == ord("n"):
                cv2.destroyAllWindows()
                return False

    return False

def extract_split_and_metadata(video_path: Path):
    print("[INFO] Extracting frames...")
    out_dir = FRAMES_DIR / video_path.stem
    left_dir = out_dir / "left"
    right_dir = out_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mid = frame.shape[1] // 2
        frame[:, :mid], frame[:, mid:] = frame[:, mid:].copy(), frame[:, :mid].copy()

        left = frame[:, :mid]
        right = frame[:, mid:]
        cv2.imwrite(left_dir / f"left_{idx:06d}.png", left)
        cv2.imwrite(right_dir / f"right_{idx:06d}.png", right)
        idx += 1

    cap.release()
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"frames_extracted": idx}, f, indent=2)
    print("[DONE]")

def main():
    out_video = next_video_path(VIDEOS_DIR)
    keep = record_video(out_video)
    if not keep:
        out_video.unlink(missing_ok=True)
        print("[INFO] Recording discarded.")
        return
    extract_split_and_metadata(out_video)

if __name__ == "__main__":
    main()



#this takes 3 frames per second

# to run:
# source .venv/bin/activate
# python capture/run_capture_and_process.py

#q stops recording, + zooms in and - arrow zooms out, and t is to toggle auto trocar tracking mode

#to run label studio : 
# label-studio start








# import cv2
# import json
# import time
# import re
# import numpy as np
# from datetime import datetime
# from pathlib import Path

# # ---------------- CONFIG ----------------
# CAM_INDEX = 0
# TARGET_W, TARGET_H = 2560, 720
# TARGET_FPS = 30
# SHOW_PREVIEW = True

# VIDEOS_DIR = Path("data/videos")
# FRAMES_DIR = Path("data/frames")

# # -------- ZOOM SETTINGS --------
# zoom_factor = 1.0
# zoom_step = 0.2
# max_zoom = 5.0
# min_zoom = 1.0

# # -------- TROCAR TRACKING --------
# auto_track = False

# # Smoothing storage
# last_center = None
# smooth_alpha = 0.7   # higher = more stable
# # ---------------------------------------


# def apply_digital_zoom(frame, zoom, center=None):
#     """Zoom into center point or frame middle."""
#     if zoom == 1.0:
#         return frame

#     h, w, _ = frame.shape
#     new_w = int(w / zoom)
#     new_h = int(h / zoom)

#     if center is None:
#         cx, cy = w // 2, h // 2
#     else:
#         cx, cy = center

#     x1 = max(0, cx - new_w // 2)
#     y1 = max(0, cy - new_h // 2)
#     x2 = min(w, x1 + new_w)
#     y2 = min(h, y1 + new_h)

#     cropped = frame[y1:y2, x1:x2]
#     return cv2.resize(cropped, (w, h))


# def detect_trocar_center(frame):
#     """
#     Detect trocar hole using Hough Circle Transform.
#     Searches only the middle ROI for stability.
#     """

#     h, w, _ = frame.shape

#     # ---- ROI: middle 60% of frame ----
#     roi_x1 = int(w * 0.2)
#     roi_x2 = int(w * 0.8)
#     roi_y1 = int(h * 0.2)
#     roi_y2 = int(h * 0.8)

#     roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, 7)

#     # Hough Circle detection
#     circles = cv2.HoughCircles(
#         gray,
#         cv2.HOUGH_GRADIENT,
#         dp=1.2,
#         minDist=50,
#         param1=120,
#         param2=30,
#         minRadius=5,
#         maxRadius=120
#     )

#     if circles is None:
#         return None

#     circles = np.uint16(np.around(circles))

#     # Take the strongest circle
#     x, y, r = circles[0][0]

#     # Convert ROI coords back to full frame coords
#     cx = int(x + roi_x1)
#     cy = int(y + roi_y1)

#     return (cx, cy), r


# def next_video_path(videos_dir: Path) -> Path:
#     videos_dir.mkdir(parents=True, exist_ok=True)
#     pattern = re.compile(r"^vid_(\d{3})\.mp4$")
#     ids = []
#     for p in videos_dir.glob("vid_*.mp4"):
#         m = pattern.match(p.name)
#         if m:
#             ids.append(int(m.group(1)))
#     next_id = (max(ids) + 1) if ids else 0
#     return videos_dir / f"vid_{next_id:03d}.mp4"


# def record_video(out_path: Path):
#     global zoom_factor, auto_track, last_center

#     cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
#     if not cap.isOpened():
#         raise RuntimeError("Could not open ZED camera.")

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
#     cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

#     actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     out = cv2.VideoWriter(
#         str(out_path),
#         cv2.VideoWriter_fourcc(*"mp4v"),
#         TARGET_FPS,
#         (actual_w, actual_h)
#     )

#     print("\n[CONTROLS]")
#     print("   +   Zoom In")
#     print("   -   Zoom Out")
#     print("   t   Toggle Trocar Tracking")
#     print("   q   Quit\n")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         zoom_center = None

#         # ---------- AUTO TROCAR MODE ----------
#         if auto_track:
#             result = detect_trocar_center(frame)

#             if result:
#                 (cx, cy), radius = result

#                 # Smooth tracking
#                 if last_center is None:
#                     last_center = (cx, cy)
#                 else:
#                     sx = int(smooth_alpha * last_center[0] + (1 - smooth_alpha) * cx)
#                     sy = int(smooth_alpha * last_center[1] + (1 - smooth_alpha) * cy)
#                     last_center = (sx, sy)

#                 zoom_center = last_center
#                 zoom_factor = 3.5

#                 # Draw detection
#                 cv2.circle(frame, last_center, radius, (0, 255, 0), 2)
#                 cv2.putText(frame, "Trocar Locked",
#                             (last_center[0] + 10, last_center[1]),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             0.8, (0, 255, 0), 2)

#             else:
#                 last_center = None
#                 cv2.putText(frame, "Searching...",
#                             (40, 60),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             1, (0, 0, 255), 2)

#         # Apply zoom
#         frame_zoomed = apply_digital_zoom(frame, zoom_factor, zoom_center)

#         out.write(frame_zoomed)

#         if SHOW_PREVIEW:
#             cv2.imshow("ZED Trocar Capture", frame_zoomed)

#             key = cv2.waitKey(1) & 0xFF

#             if key == ord("q"):
#                 break

#             elif key == ord("t"):
#                 auto_track = not auto_track
#                 print("[AUTO TRACK]", "ON" if auto_track else "OFF")

#             elif key == ord("+") or key == ord("="):
#                 zoom_factor = min(max_zoom, zoom_factor + zoom_step)
#                 print(f"[ZOOM] {zoom_factor:.1f}x")

#             elif key == ord("-"):
#                 zoom_factor = max(min_zoom, zoom_factor - zoom_step)
#                 print(f"[ZOOM] {zoom_factor:.1f}x")

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()


# def extract_split_and_metadata(video_path: Path):
#     video_name = video_path.stem
#     out_dir = FRAMES_DIR / video_name
#     left_dir = out_dir / "left"
#     right_dir = out_dir / "right"

#     left_dir.mkdir(parents=True, exist_ok=True)
#     right_dir.mkdir(parents=True, exist_ok=True)

#     cap = cv2.VideoCapture(str(video_path))

#     extracted = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         mid = frame.shape[1] // 2
#         left = frame[:, :mid]
#         right = frame[:, mid:]

#         cv2.imwrite(str(left_dir / f"left_{extracted:06d}.png"), left)
#         cv2.imwrite(str(right_dir / f"right_{extracted:06d}.png"), right)
#         extracted += 1

#     cap.release()

#     with open(out_dir / "metadata.json", "w") as f:
#         json.dump({"frames_extracted": extracted}, f, indent=2)

#     return extracted


# def main():
#     out_video = next_video_path(VIDEOS_DIR)

#     record_video(out_video)

#     print("\n[INFO] Extracting frames...")
#     extracted = extract_split_and_metadata(out_video)

#     print("\n[DONE]")
#     print("Video:", out_video.resolve())
#     print("Frames extracted:", extracted)


# if __name__ == "__main__":
#     main()


# to run:
# source .venv/bin/activate
# python capture/run_capture_and_process.py

#q stops recording, + zooms in and - arrow zooms out, and t is to toggle auto trocar tracking mode

#to run label studio : 
# label-studio start

#edit code to take 2/3 frames per second 






















# import cv2
# import json
# import time
# import re
# from datetime import datetime
# from pathlib import Path

# # ---------------- CONFIG ----------------
# CAM_INDEX = 0
# TARGET_W, TARGET_H = 2560, 720
# TARGET_FPS = 30
# SHOW_PREVIEW = True

# VIDEOS_DIR = Path("data/videos")
# FRAMES_DIR = Path("data/frames")
# # ---------------------------------------

# def next_video_path(videos_dir: Path) -> Path:
#     videos_dir.mkdir(parents=True, exist_ok=True)
#     pattern = re.compile(r"^vid_(\d{3})\.mp4$")
#     ids = []
#     for p in videos_dir.glob("vid_*.mp4"):
#         m = pattern.match(p.name)
#         if m:
#             ids.append(int(m.group(1)))
#     next_id = (max(ids) + 1) if ids else 0
#     return videos_dir / f"vid_{next_id:03d}.mp4"

# def record_video(out_path: Path) -> dict:
#     cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
#     if not cap.isOpened():
#         raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
#     cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

#     actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS)

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(str(out_path), fourcc, TARGET_FPS, (actual_w, actual_h))

#     print(f"[INFO] Recording to: {out_path.resolve()}")
#     print(f"[INFO] Camera {CAM_INDEX}: {actual_w}x{actual_h} @ {actual_fps:.2f} fps")
#     print("[INFO] Press 'q' in the preview window to stop.")

#     start = time.time()
#     frames = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("[WARN] Failed to read frame.")
#             break

#         out.write(frame)
#         frames += 1

#         if SHOW_PREVIEW:
#             cv2.imshow("ZED index 0 (SBS)", frame)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#     elapsed = time.time() - start

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     return {
#         "video_file": out_path.name,
#         "video_path": str(out_path),
#         "duration_seconds": elapsed,
#         "frames_written_est": frames,
#         "capture_resolution": [actual_w, actual_h],
#         "capture_fps_reported": actual_fps,
#     }

# def extract_split_and_metadata(video_path: Path) -> dict:
#     video_name = video_path.stem
#     out_dir = FRAMES_DIR / video_name
#     left_dir = out_dir / "left"
#     right_dir = out_dir / "right"
#     left_dir.mkdir(parents=True, exist_ok=True)
#     right_dir.mkdir(parents=True, exist_ok=True)

#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         raise RuntimeError(f"Could not open video: {video_path}")

#     fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
#     frame_count_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

#     extracted = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         h, w, _ = frame.shape
#         mid = w // 2
#         left = frame[:, :mid]
#         right = frame[:, mid:]

#         cv2.imwrite(str(left_dir / f"left_{extracted:06d}.png"), left)
#         cv2.imwrite(str(right_dir / f"right_{extracted:06d}.png"), right)
#         extracted += 1

#     cap.release()

#     meta = {
#         "video_file": video_path.name,
#         "video_stem": video_name,
#         "processed_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
#         "opencv_version": cv2.__version__,
#         "input_video_properties": {
#             "fps_reported": fps,
#             "frame_count_reported": frame_count_reported,
#             "width_reported": width,
#             "height_reported": height,
#         },
#         "extraction": {
#             "frames_extracted": extracted,
#             "split_mode": "side_by_side",
#             "output_root": str(out_dir),
#             "left_dir": str(left_dir),
#             "right_dir": str(right_dir),
#             "filename_pattern_left": "left_%06d.png",
#             "filename_pattern_right": "right_%06d.png",
#         },
#     }

#     with open(out_dir / "metadata.json", "w") as f:
#         json.dump(meta, f, indent=2)

#     return {"frames_extracted": extracted, "frames_dir": str(out_dir)}

# def main():
#     out_video = next_video_path(VIDEOS_DIR)
#     capture_info = record_video(out_video)

#     print("[INFO] Recording complete. Starting extraction...")
#     process_info = extract_split_and_metadata(out_video)

#     print("[DONE]")
#     print(f"  Video:  {out_video.resolve()}")
#     print(f"  Frames: {Path(process_info['frames_dir']).resolve()}")
#     print(f"  Extracted frames: {process_info['frames_extracted']}")

# if __name__ == "__main__":
#     main()


#to run:
# source .venv/bin/activate
# python capture/run_capture_and_process.py

#takes 30 frames per second