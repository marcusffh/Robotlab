# find_landmark_min.py  — minimal ArUco chase
import time, cv2, numpy as np
from picamera2 import Picamera2
import robot

# --- tiny config (tune if needed) ---
IMG_W, IMG_H = 640, 480
CENTER_TOL_PIX = 40      # how close to center before driving forward
STOP_PIX = 140           # stop when marker height (px) >= this
TURN_PWR, FWD_PWR = 40, 64   # per robot.py: >=40 except 0

# --- init robot + camera ---
arlo = robot.Robot()
cam = Picamera2()
cfg = cam.create_video_configuration({"size": (IMG_W, IMG_H), "format": "RGB888"})
cam.configure(cfg); cam.start(); time.sleep(1)

# --- aruco setup ---
aruco = cv2.aruco
DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
PARAMS = aruco.DetectorParameters_create()

def step(cmd, dt=0.12):
    # cmd: 'L','R','F' for left/right/forward step
    if cmd == 'L': arlo.go_diff(TURN_PWR, TURN_PWR, 0, 1)
    if cmd == 'R': arlo.go_diff(TURN_PWR, TURN_PWR, 1, 0)
    if cmd == 'F': arlo.go_diff(FWD_PWR, FWD_PWR, 1, 1)
    time.sleep(dt); arlo.stop()

try:
    while True:
        frame = cam.capture_array("main")
        corners, ids, _ = aruco.detectMarkers(frame, DICT, parameters=PARAMS)

        if ids is None or len(ids) == 0:
            step('L')             # slow spin until we see a marker
            continue

        c = corners[0][0]         # use the first/strongest marker
        cx = int(np.mean(c[:,0]))
        mid = IMG_W // 2
        # estimate marker height in pixels from an edge (top-left -> bottom-left)
        hpx = int(np.linalg.norm(c[0] - c[3]))

        # 1) center horizontally
        if abs(cx - mid) > CENTER_TOL_PIX:
            step('L' if cx < mid else 'R')
            continue

        # 2) drive forward until marker looks "big enough", then stop
        if hpx < STOP_PIX:
            step('F', 0.20 if hpx < STOP_PIX*0.8 else 0.12)
        else:
            print(f"Done: centered and close (height≈{hpx}px).")
            break
finally:
    arlo.stop()
