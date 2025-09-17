import cv2, time
import numpy as np
from picamera2 import Picamera2
import robot

# Init robot + camera
arlo = robot.Robot()
cam = Picamera2()
cfg = cam.create_video_configuration({"size": (640,480), "format":"RGB888"})
cam.configure(cfg); cam.start(); time.sleep(1)

# ArUco setup
aruco = cv2.aruco
dict6x6 = aruco.Dictionary_get(aruco.DICT_6X6_250)
params = aruco.DetectorParameters_create()

while True:
    frame = cam.capture_array("main")
    corners, ids, _ = aruco.detectMarkers(frame, dict6x6, parameters=params)

    if ids is None:  # spin until marker seen
        arlo.go_diff(50, 50, 0, 1); time.sleep(0.1); arlo.stop()
        continue

    # Marker center
    c = corners[0][0]
    cx = int(np.mean(c[:,0])); mid = frame.shape[1]//2
    if abs(cx-mid) > 40:  # rotate until centered
        if cx < mid: arlo.go_diff(40,40,0,1)
        else:        arlo.go_diff(40,40,1,0)
        time.sleep(0.1); arlo.stop()
    else:  # drive forward
        arlo.go_diff(60,60,1,1); time.sleep(0.3); arlo.stop()
        break
