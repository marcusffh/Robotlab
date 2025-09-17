import cv2
import cv2.aruco as aruco
import numpy as np
import time
from Exercise1.CalibratedRobot import CalibratedRobot


# Initialize robot
calArlo = CalibratedRobot()

# Camera parameters (replace fx, fy, cx, cy with your calibration results)
focal_length = 1275  # example from task 1
camera_matrix = np.array([[focal_length, 0, 512],
                          [0, focal_length, 384],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))  # assume no distortion if not calibrated

# ArUco dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

# Open camera
cap = cv2.VideoCapture(0)  # or gstreamer pipeline on robot

def search_and_drive():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            # Found marker
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, 100, camera_matrix, dist_coeffs)  # 100mm marker size

            # Take first detected marker
            tvec = tvecs[0][0]  # (x, y, z) in mm

            print("Marker at:", tvec)

            # Determine angle: if marker is left or right
            angle = np.degrees(np.arctan2(tvec[0], tvec[2]))
            dist = tvec[2] / 1000.0  # convert mm → meters

            # Rotate to face marker
            calArlo.turn_angle(angle)

            # Drive forward some distance
            calArlo.drive_distance(min(dist, 0.5))  # step 0.5m max

            # If close enough, stop
            if dist < 0.2:
                print("Reached landmark!")
                break
        else:
            # Not visible → rotate slowly
            calArlo.drive(50, 50, calArlo.BACKWARD, calArlo.FORWARD)
            time.sleep(0.2)
            calArlo.stop()

try:
    search_and_drive()
finally:
    cap.release()
    calArlo.stop()
    cv2.destroyAllWindows()
