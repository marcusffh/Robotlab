import cv2
import numpy as np
from picamera2 import Picamera2



class CameraUtils:
    """  A very simple utility class used to manage the rapberry Pi
    camera on the Arlo robot used in this course. Please notice that
    it converts the captured frames to the BGR color format used by
    OpenCV.
    """
    ## Camera calibration
    def __init__(self, width=640, height=480, fx=1275, fy=1275, cx=None, cy=None):
        self.picam2 = None
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx if cx is not None else width / 2 ## 
        self.cy = cy if cy is not None else height / 2 ##
        self.dist = np.zeros((5, 1))  # assume no distortion

    @property
    def camera_matrix(self):
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0,     0,     1]
            ], dtype=np.float32)

    def start_camera(self, width: int = 640, height: int = 480, fps: int = 30):
        """Start the PiCamera2."""

        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()

    def get_frame(self):
        """Capture one frame as a BGR image (for OpenCV)."""
        if self.picam2 is None:
            raise RuntimeError("Camera not started. Call start_camera() first.")
        frame_rgb = self.picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return True, frame_bgr

    def stop_camera(self):
        """Stop the PiCamera2."""
        if self.picam2 is not None:
            self.picam2.stop()
            self.picam2 = None







