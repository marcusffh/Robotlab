import time
import numpy as np
from RobotUtils.CameraUtils import CameraUtils, ArucoUtils
import cv2
from RobotUtils.CalibratedRobot import CalibratedRobot
import matplotlib.pyplot as plt

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
            print(f"angle: {angle}")
        
            calArlo.turn_angle(angle)
        
            if not isDriving and marker_id != last_id:
                isDriving = True
                calArlo.drive_distance(dist)
            
            if dist <= 0:
                last_id = marker_id
                isDriving = False
        else:
            calArlo.turn_angle(15)
            time.sleep(0.2)

def drive_to_landmark_steps():
    step_distance = 0.4
    while True:
        frame = cam.get_frame()
        corners, ids = aruco.detect_markers(frame)
        
        if ids is not None:
            rvecs, tvecs = aruco.estimate_pose(corners, cam.camera_matrix)
            tvec = tvecs[0][0]

            dist = aruco.compute_distance_to_marker(tvec)
            angle = aruco.compute_rotation_to_marker(tvec)
            print(f"distance: {dist}")
            print(f"angle: {angle}")

            calArlo.turn_angle(angle)

            while dist > 0.1:  
                drive_step = min(step_distance, dist)
                print(f"Drove distance: {drive_step}")
                calArlo.drive_distance(drive_step)
                time.sleep(0.05)

                frame = cam.get_frame()
                corners, ids = aruco.detect_markers(frame)
                if ids is None:
                    print("Lost marker")
                    break
                tvec = aruco.estimate_pose(corners, cam.camera_matrix)[1][0][0]
                dist = aruco.compute_distance_to_marker(tvec)
                angle = aruco.compute_rotation_to_marker(tvec)
                calArlo.turn_angle(angle)
        else:
            calArlo.turn_angle(15)
            time.sleep(0.4)

try:
    drive_to_landmark_steps()
finally:
    calArlo.stop()
    cam.stop_camera()