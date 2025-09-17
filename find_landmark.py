#!/usr/bin/env python3
"""
find_landmark.py — Rotate, search for an ArUco marker, align, and approach.

Uses:
  - Exercise1.CalibratedRobot for movement (with calibration)
  - OpenCV for ArUco detection
"""

import time
import cv2
import numpy as np
from Exercise1.CalibratedRobot import CalibratedRobot

# ===== PARAMETERS =====
F_PX       = 1290.0     # focal length from your calibration
MARKER_MM  = 140.0      # marker size (mm)
STOP_AT_MM = 450.0      # stop distance
PX_TOL     = 20         # pixel tolerance for centering
Kp         = 0.0015     # steering gain

SEARCH_PWR = 60
DRIVE_PWR  = 62
SPIN_ANGLE = 10         # degrees per search step
ALIGN_ANGLE = 5         # degrees per alignment step
STEP_MS    = 320        # ms per forward step
# =======================


# --- Marker detection ---
def detect_marker(frame, target_id=None):
    """Return (cx, x_px, w) or None if no marker detected."""
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
        pts = c.reshape(-1, 2)
        per = sum(np.linalg.norm(pts[j] - pts[(j + 1) % 4]) for j in range(4))
        if best is None or per > best[0]:
            best = (per, pts)
    if best is None:
        return None

    _, pts = best
    TL, TR, BR, BL = pts
    x_px = 0.5 * (np.linalg.norm(TL - BL) + np.linalg.norm(TR - BR))
    cx = np.mean(pts[:, 0])
    w = frame.shape[1]
    return cx, x_px, w


def estimate_Z_mm(x_px):
    return (F_PX * MARKER_MM) / max(x_px, 1e-6)


# --- Main ---
def main():
    rb = CalibratedRobot()

    # Force V4L2 backend, avoid GStreamer
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not cap.isOpened():
        raise RuntimeError("Camera failed to open. Check /dev/video0 and drivers.")

    state = "SEARCH"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            det = detect_marker(frame)

            if state == "SEARCH":
                if det is None:
                    rb.turn_angle(SPIN_ANGLE)  # rotate left
                    continue
                state = "ALIGN"
                continue

            if det is None:
                state = "SEARCH"
                continue

            cx, x_px, w = det
            Z = estimate_Z_mm(x_px)
            err = cx - w / 2

            if state == "ALIGN":
                if abs(err) <= PX_TOL:
                    state = "APPROACH"
                else:
                    rb.turn_angle(ALIGN_ANGLE if err > 0 else -ALIGN_ANGLE)
                continue

            if state == "APPROACH":
                if Z <= STOP_AT_MM:
                    rb.stop()
                    print(f"STOP at {Z:.0f} mm")
                    break
                # proportional steering
                bias = max(min(Kp * err, 0.5), -0.5)
                l = DRIVE_PWR * (1 - bias)
                r = DRIVE_PWR * (1 + bias)
                rb.drive(l, r, rb.FORWARD, rb.FORWARD)
                time.sleep(STEP_MS / 1000.0)
                rb.stop()

    except KeyboardInterrupt:
        print("\n[ABORT] Ctrl+C")
        rb.stop()
    finally:
        cap.release()
        rb.stop()


if __name__ == "__main__":
    main()
