import cv2

CAM_INDEX = 0  # change to 1 or 2 to test
cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

print("Press q to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow(f"Camera {CAM_INDEX}", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

#python capture/preview_camera.py
