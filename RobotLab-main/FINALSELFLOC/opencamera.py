import cv2
from camera import Camera  # make sure Camera.py is in the same folder or in PYTHONPATH
import time

def main():
    # Initialize camera
    cam = Camera(0, robottype='macbookpro', useCaptureThread=False)  # change robottype if needed
    WIN_NAME = "Camera Test"
    cv2.namedWindow(WIN_NAME)
    cv2.moveWindow(WIN_NAME, 50, 50)

    try:
        while True:
            # Grab next frame
            frame = cam.get_next_frame()

            # Show frame
            cv2.imshow(WIN_NAME, frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Small sleep to avoid overloading CPU
            time.sleep(0.02)

    finally:
        cv2.destroyAllWindows()
        cam.terminateCaptureThread()

if __name__ == "__main__":
    main()
