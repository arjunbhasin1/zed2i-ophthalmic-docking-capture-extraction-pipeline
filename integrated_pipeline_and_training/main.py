import cv2
from vision.camera import Camera
from vision.detector import Detector
from robot.meca500 import Meca500
from utils.control import compute_error, clip
from config import *

def main():
    cam = Camera()
    detector = Detector()
    robot = Meca500()

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        needle, target, vis = detector.detect(frame)

        if needle and target:
            dx, dy = compute_error(needle, target)

            print(f"dx={dx}, dy={dy}")

            if abs(dx) < THRESHOLD and abs(dy) < THRESHOLD:
                print("Aligned → Insert")

                robot.move_relative(0, 0, Z_INSERT)
                robot.move_relative(0, 0, -Z_INSERT)
                break

            move_x = clip(dx * KX, MAX_STEP)
            move_y = clip(dy * KY, MAX_STEP)

            robot.move_relative(move_x, move_y, 0)

        cv2.imshow("Detection", vis)

        if cv2.waitKey(1) == 27:
            break

    cam.release()
    robot.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()