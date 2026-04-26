import cv2
from config import CAMERA_ID

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_ID)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()