import cv2
import numpy as np
from picamera2 import Picamera2 ## should only be downloaded onto the raspberry pi, will not run on your own pc



class CameraUtils:
    """  A very simple utility class used to manage the rapberry Pi
    camera on the Arlo robot used in this course. Please notice that
    it converts the captured frames to the BGR color format used by
    OpenCV.
    """
    ## Camera calibration
    def __init__(self, width=1640, height=1232, fx=1360, fy=1360, cx=None, cy=None, fps = 30):
        self.picam2 = None
        self.width = width #resultion
        self.height = height #resulution
        self.fps = fps 
        self.fx = fx #focal length
        self.fy = fy # --||--
        self.cx = cx if cx is not None else width / 2 ## image sensor
        self.cy = cy if cy is not None else height / 2 ## --||--
        self.dist = np.zeros((5, 1))  # assume no distortion
        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0,     0,     1]
            ], dtype=np.float32)

    def start_camera(self):
        """Start the PiCamera2."""
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
          # Apply fixed fps
        frame_time = int(1e6 / self.fps)
        config["controls"]["FrameDurationLimits"] = (frame_time, frame_time)
        self.picam2.configure(config)
        self.picam2.start()

    def get_frame(self):
        """Capture one frame as a BGR image (for OpenCV)."""
        if self.picam2 is None:
            raise RuntimeError("Camera not started. Call start_camera() first.")
        frame_rgb = self.picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return frame_bgr

    def stop_camera(self):
        """Stop the PiCamera2."""
        if self.picam2 is not None:
            self.picam2.stop()
            self.picam2 = None

class ArucoUtils:
    """ Simple utility for detecting ArUco markers and estimating pose. """
    def __init__(self, marker_length=0.145): ## real life height of the aruco
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250) # what kind of arUco markers are we looking for
        self.aruco_params = cv2.aruco.DetectorParameters_create() # Sets parameters for detection algorithm
        self.marker_length = marker_length # sets real life length of aruco marke, 0.14 m in our case

    def detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert image to greyscale
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )# The actual algorithm
        return corners, ids #(coordinate of four corners, id of aruco marker)

    def estimate_pose(self, corners, camera_matrix, dist_coeffs=None):
        if dist_coeffs is None:
            dist_coeffs = np.zeros((5, 1)) # lens distortion, set to zero
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, camera_matrix, dist_coeffs
        ) #Computes 3d pose of of each detected marker
        return rvecs, tvecs #(Rotation vector, translation vector)
    
    def compute_distance_to_marker(self, tvec):
        dist = cv2.norm(tvec)
        return dist
        
    
    def compute_rotation_to_marker(self, tvec):
            z_axis = np.array([0, 0, 1])
            cos_beta = np.dot(tvec, z_axis) / np.linalg.norm(tvec)
            
            beta = np.arccos(cos_beta)
            return beta
        