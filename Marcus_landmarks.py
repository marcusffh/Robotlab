# find_landmark_search_then_drive_updated.py
import time
import numpy as np
from Robotutils.CameraDetection_util import CameraUtils, ArucoUtils
import Robotutils.robot as robot
import cv2
from Robotutils.CalibratedRobot import CalibratedRobot


IMG_W, IMG_H, FPS  = 960, 720, 10


# ==== Init Robot + Camera + Aruco ====
calArlo = CalibratedRobot()
cam = CameraUtils(width=IMG_W, height=IMG_H)
cam.start_camera(width=IMG_W, height=IMG_H, fps=FPS)
aruco = ArucoUtils()

def drive_to_landmark():
    isDriving = False
    STOP_BUFFER = 0.3
    last_id = None

    while True:
        frame = cam.get_frame()
        corners, ids = aruco.detect_markers(frame)
        if ids is not None:
            ids = ids.flatten()
            for i, marker_id in enumerate(ids):
                if marker_id == last_id:
                    continue 
                print(f"id found: {marker_id}")
                rvecs, tvecs = aruco.estimate_pose(corners, cam.camera_matrix)
                tvec = tvecs[i][0]

                dist = (aruco.compute_distance_to_marker(tvec)) - STOP_BUFFER
                dist = max(0,dist)
                angle = aruco.compute_rotation_to_marker(tvec)
            
                calArlo.turn_angle(angle)
            
                if not isDriving:
                    isDriving = True
                    calArlo.drive_distance(dist)
                
                if dist <= 0.05:
                    last_id = marker_id
                    isDriving = False
                    break
        else:
            calArlo.drive(20, 20, calArlo.BACKWARD, calArlo.FORWARD)
            time.sleep(0.35)
            calArlo.stop()
            
drive_to_landmark()
