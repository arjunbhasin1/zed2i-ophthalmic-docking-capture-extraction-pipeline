from ultralytics import YOLO
from config import MODEL_PATH

class Detector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)

    def detect(self, frame):
        results = self.model(frame)[0]

        needle = None
        target = None

        for box in results.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0]

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if cls == 0:
                needle = (cx, cy)
            elif cls == 1:
                target = (cx, cy)

        return needle, target, results.plot()