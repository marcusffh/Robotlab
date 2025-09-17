# map_landmarks_rotate.py
# Maps all visible ArUco markers in front of the robot for a few headings.
# Uses: PiCamera, OpenCV ArUco, your CalibratedRobot methods only.

import cv2
import numpy as np
import time
from picamera2 import PiCamera2
from picamera.array import PiRGBArray
from Exercise1.CalibratedRobot import CalibratedRobot  # adjust path if needed

# --- From your setup / slides ---
F_PIX = 1275.0          # focal length [pixels]
MARKER_LEN_M = 0.14     # 140 mm -> meters

# Camera setup
RES_W, RES_H = 640, 480
camera = PiCamera2()
camera.resolution = (RES_W, RES_H)
camera.framerate = 30
raw = PiRGBArray(camera, size=(RES_W, RES_H))
time.sleep(0.2)  # warmup

# Intrinsics: assume principal point at image centre (slides allow this)
CX, CY = RES_W/2.0, RES_H/2.0  # :contentReference[oaicite:3]{index=3}
K = np.array([[F_PIX, 0.0,   CX],
              [0.0,   F_PIX, CY],
              [0.0,   0.0,   1.0]], dtype=np.float64)
dist = np.zeros((5, 1), dtype=np.float64)

# ArUco (per slides)
aruco = cv2.aruco
DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
PARAMS = aruco.DetectorParameters_create()  # :contentReference[oaicite:4]{index=4}

# Robot (we ONLY use existing methods)
bot = CalibratedRobot()

def scan_once():
    """Capture one frame, detect all markers, return list of (id, x, z, dist)."""
    camera.capture(raw, format="bgr", use_video_port=True)
    frame = raw.array
    raw.truncate(0); raw.seek(0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, DICT, parameters=PARAMS)  # :contentReference[oaicite:5]{index=5}

    results = []
    if ids is not None and len(ids) > 0:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, MARKER_LEN_M, K, dist
        )  # returns tvecs in camera coords (meters) :contentReference[oaicite:6]{index=6}

        for i in range(len(ids)):
            t = tvecs[i, 0, :]                # (tx, ty, tz)
            x, z = float(t[0]), float(t[2])   # 2D map in camera frame
            d = float(np.linalg.norm(t))      # optional full distance
            results.append((int(ids[i][0]), x, z, d))
    return sorted(results, key=lambda r: r[3])  # nearest first

try:
    headings = [-60, -30, 0, 30, 60]   # small loop of turns (deg)
    for a in headings:
        if a != 0:
            bot.turn_angle(a)          # existing API: relative turn

        pts = scan_once()
        print(f"\nHeading {a:+} deg — landmarks (id, x[m], z[m], dist[m]):")
        if not pts:
            print("  (none)")
        else:
            for rec in pts:
                print(f"  {rec}")

    bot.stop()

finally:
    bot.stop()
    camera.close()
