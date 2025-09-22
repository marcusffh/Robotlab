# find_landmark_search_then_drive_updated.py
import time
import numpy as np
from Robotutils.CameraDetection_util import CameraUtils, ArucoUtils
import Robotutils.robot as robot
import cv2
from Robotutils.CalibratedRobot import CalibratedRobot


# ==== Init Robot + Camera + Aruco ====
calArlo = CalibratedRobot()
cam = CameraUtils()
cam.start_camera()
aruco = ArucoUtils()

def drive_to_landmark():
    isDriving = False
    last_id = None

    while True:
        frame = cam.get_frame()
        corners, ids = aruco.detect_markers(frame)
        if ids is not None:
            marker_id = int(ids[0][0])
            print(f"id found: {marker_id}")
            rvecs, tvecs = aruco.estimate_pose(corners, cam.camera_matrix)
            tvec = tvecs[0][0]

            dist = aruco.compute_distance_to_marker(tvec)
            angle = aruco.compute_rotation_to_marker(tvec)
            print(f"distance: {dist}")
        
            calArlo.turn_angle(angle)
        
            if not isDriving and marker_id != last_id:
                isDriving = True
                calArlo.drive_distance(dist)
            
            if dist <= 0:
                last_id = marker_id
                isDriving = False
        else:
            calArlo.turn_angle(15)
            calArlo.stop()
           # time.sleep(1)

try:
    drive_to_landmark()
finally:
    calArlo.stop()
    cam.stop_camera()
