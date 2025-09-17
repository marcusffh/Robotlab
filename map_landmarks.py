# map_landmarks_rotate_cv2.py
# Maps visible ArUco markers at a few headings using ONLY OpenCV for camera I/O.

import cv2
import numpy as np
import time
from Exercise1.CalibratedRobot import CalibratedRobot  # adjust path if needed

# --- From your setup / slides ---
F_PIX = 1275.0          # focal length [pixels]
MARKER_LEN_M = 0.14     # 140 mm -> meters

# -------- Camera setup via OpenCV --------
RES_W, RES_H = 640, 480
# If V4L2 is needed explicitly, use: cv2.VideoCapture(0, cv2.CAP_V4L2)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RES_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_H)
cap.set(cv2.CAP_PROP_FPS, 30)

# Confirm actual size (some drivers ignore set)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or RES_W)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or RES_H)
RES_W, RES_H = w, h

# Intrinsics: assume principal point at image centre (slides allow this)
CX, CY = RES_W/2.0, RES_H/2.0
K = np.array([[F_PIX, 0.0,   CX],
              [0.0,   F_PIX, CY],
              [0.0,   0.0,   1.0]], dtype=np.float64)
dist = np.zeros((5, 1), dtype=np.float64)

# ArUco (per slides)
aruco = cv2.aruco
DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
PARAMS = aruco.DetectorParameters_create()

# Robot (we ONLY use existing methods)
bot = CalibratedRobot()

def scan_landmarks():
    """Capture one frame, detect all markers, return list of (id, x, z, dist)."""
    ok, frame = cap.read()
    if not ok or frame is None:
        return []  # no frame -> nothing detected

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, DICT, parameters=PARAMS)

    results = []
    while True:
        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, MARKER_LEN_M, K, dist
            )  # returns tvecs in camera coords (meters) :contentReference[oaicite:6]{index=6}

            for i in range(len(ids)):
                t = tvecs[i, 0, :]                # (tx, ty, tz)
                x, z = float(t[0]), float(t[2])   # 2D map in camera frame
                d = float(np.linalg.norm(t))      # optional full distance
                results.append((int(ids[i][0]), x, z, d))
        else:
            print("Searching for marker...")
            bot.drive(50, 50, bot.BACKWARD, bot.FORWARD)
            time.sleep(0.2)
            bot.stop()
        return sorted(results, key=lambda r: r[3])  # nearest first

try:
    pts = scan_once()
    print(f"\n - landmarks (id, x[m], z[m], dist[m]):")
    if not pts:
        print(" (none)")
    else:
        for rec in pts:
            print(f"  {rec}")

    bot.stop()

finally:
    bot.stop()
    cap.release()
    cv2.destroyAllWindows()
