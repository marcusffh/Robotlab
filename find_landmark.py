#!/usr/bin/env python3
"""
find_landmark_arlo.py — Use CalibratedRobot + OpenCV ArUco.

Flow:
  1) Rotate in pulses until marker detected.
  2) Align so marker is centered in image.
  3) Drive toward it in steps, correcting heading.
  4) Stop when within STOP_AT_MM.
"""

import time
import cv2
import numpy as np

from Exercise1.CalibratedRobot import CalibratedRobot


# ===== PARAMETERS =====
F_PX       = 1290.0     # focal length from your calibration
MARKER_MM  = 140.0      # marker size (mm)
STOP_AT_MM = 450.0
PX_TOL     = 20         # pixels tolerance for center
Kp         = 0.0015     # steering gain

SEARCH_PWR = 60
DRIVE_PWR  = 62
SPIN_MS    = 120
STEP_MS    = 320
# =======================

def detect_marker(frame, target_id=None):
    """Return (cx, x_px, w) or None."""
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    params = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(frame, dictionary, parameters=params)
    if ids is None: 
        return None

    best = None
    for c, i in zip(corners, ids.flatten()):
        if target_id is not None and i != target_id:
            continue
        pts = c.reshape(-1,2)
        per = sum(np.linalg.norm(pts[j]-pts[(j+1)%4]) for j in range(4))
        if best is None or per > best[0]:
            best = (per, pts)
    if best is None: return None

    _, pts = best
    TL, TR, BR, BL = pts
    x_px = 0.5*(np.linalg.norm(TL-BL)+np.linalg.norm(TR-BR))
    cx = np.mean(pts[:,0])
    w = frame.shape[1]
    return cx, x_px, w

def estimate_Z_mm(x_px):
    return (F_PX * MARKER_MM) / max(x_px, 1e-6)

def main():
    rb = CalibratedRobot()
    cap = cv2.VideoCapture(0)   # adjust index if needed

    state = "SEARCH"
    try:
        while True:
            ok, frame = cap.read()
            if not ok: 
                continue
            det = detect_marker(frame)

            if state == "SEARCH":
                if det is None:
                    rb.turn_angle(10)   # small left pulse
                    continue
                state = "ALIGN"
                continue

            if det is None:
                state = "SEARCH"
                continue

            cx, x_px, w = det
            Z = estimate_Z_mm(x_px)
            err = cx - w/2

            if state == "ALIGN":
                if abs(err) <= PX_TOL:
                    state = "APPROACH"
                else:
                    rb.turn_angle(5 if err>0 else -5)
                continue

            if state == "APPROACH":
                if Z <= STOP_AT_MM:
                    rb.stop()
                    print(f"STOP at {Z:.0f} mm")
                    break
                bias = max(min(Kp*err,0.5),-0.5)
                # steer by adjusting left/right power
                l = DRIVE_PWR*(1-bias)
                r = DRIVE_PWR*(1+bias)
                rb.drive(l,r,rb.FORWARD,rb.FORWARD)
                time.sleep(STEP_MS/1000.0)
                rb.stop()

    except KeyboardInterrupt:
        rb.stop()
    finally:
        cap.release()
        rb.stop()

if __name__ == "__main__":
    main()
