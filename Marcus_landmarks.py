# find_landmark_search_then_drive_updated.py
import time
import numpy as np
from Robotutils.CameraDetection_util import CameraUtils, ArucoUtils
import Robotutils.robot as robot
import cv2
from Robotutils.CalibratedRobot import CalibratedRobot


IMG_W, IMG_H, FPS  = 960, 720, 10


# ==== Init Robot + Camera + Aruco ====
calArlo = CalibratedRobot
cam = CameraUtils(width=IMG_W, height=IMG_H)
cam.start_camera(width=IMG_W, height=IMG_H, fps=FPS)
aruco = ArucoUtils()

def drive_to_landmark():
    isDriving = False
    STOP_BUFFER = 0.2

    while True:
        frame = cam.get_frame()
        corners, ids = aruco.detect_markers(frame)
        if ids is not None:
            print(f"id found: {ids}")
            rvecs, tvecs = aruco.estimate_pose(corners, cam.camera_matrix)
            tvec = tvecs[0]
                        
            dist = aruco.compute_distance_to_marker(tvec)
            angle = aruco.compute_rotation_to_marker(tvec)
            
            calArlo.turn_angle(angle)
            
            if not isDriving:
                isDriving = True
                calArlo.drive_distance(dist)
                
            if dist <= 0:
                print("Reached landmark!")
                calArlo.stop()
                driving = False
                break
        else:
            print("finished")
            calArlo.drive(50, 50, calArlo.BACKWARD, calArlo.FORWARD)
            time.sleep(0.2)
            calArlo.stop()
            
drive_to_landmark()
