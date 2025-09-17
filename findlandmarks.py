#!/usr/bin/env python3
"""
find_landmarks.py
-----------------
Rotate to find an ArUco marker, align to center (pixel-based), and step toward it.
Uses your CalibratedRobot (turn_angle, drive_distance, stop) + the detector that
already works for you (from aruco_utils.detect_one).
"""

import time
import numpy as np
import cv2

from exercise1.CalibratedRobot import CalibratedRobot  # ensure exercise1/__init__.py exists
from aruco_utils import ArucoUtils

# ---- Camera / Marker (pixel-only; same idea as your working script) ----
F_PX      = 1275.0   # if you later want to compute Z_mm = f*X/x
MARKER_MM = 140.0
TARGET_ID = None

# ---- Simple tuning (angles & steps; keep it minimal) ----
SEARCH_STEP_DEG = 15.0     # rotate a bit, stop, check
SEARCH_SLEEP_S  = 0.10

PX_TOL          = 28       # deadband around image center (pixels)
PX_KP_DEG_PER_PX = 0.08    # map pixel error -> small turn (deg)
MAX_ALIGN_STEP_DEG = 18.0

STEP_FWD_M      = 0.25     # forward step length
STEP_FWD_MIN_M  = 0.10
STOP_SIDE_PX    = 240      # stop when tag looks this big (proxy for distance)
LOST_LIMIT      = 8
LOOP_SLEEP_S    = 0.05

def estimate_Z_mm(x_px, f_px=F_PX, X_mm=MARKER_MM):
    return (f_px * X_mm) / max(x_px, 1e-6)

def main():
    bot = CalibratedRobot()
    aru = ArucoUtils(res=(960, 720), fps=30)  # matches your working cam settings
    aru.start_camera()

    state = "SEARCH"
    lost = 0

    try:
        while True:
            ok, frame = aru.read()
            if not ok:
                bot.stop()
                time.sleep(0.05)
                continue

            det = aru.detect_one(frame, restrict_id=TARGET_ID)

            if state == "SEARCH":
                if det is None:
                    ArucoUtils.rotate_step(bot, SEARCH_STEP_DEG)
                    time.sleep(SEARCH_SLEEP_S)
                    continue
                state = "ALIGN"
                lost = 0
                continue

            if det is None:
                lost += 1
                if lost >= LOST_LIMIT:
                    state = "SEARCH"
                time.sleep(LOOP_SLEEP_S)
                continue

            # Got a detection — compute pixel error and apparent size
            cx, w = det["cx"], det["w"]
            err_px = cx - (w * 0.5)
            x_px   = det["x_px"]

            if state == "ALIGN":
                if abs(err_px) <= PX_TOL:
                    state = "APPROACH"
                else:
                    step_deg = float(np.clip(err_px * PX_KP_DEG_PER_PX, -MAX_ALIGN_STEP_DEG, MAX_ALIGN_STEP_DEG))
                    ArucoUtils.rotate_step(bot, step_deg)
                    time.sleep(SEARCH_SLEEP_S)
                continue

            if state == "APPROACH":
                # Stop condition in pixel-mode: when the tag looks big enough
                if x_px >= STOP_SIDE_PX:
                    bot.stop()
                    print("Arrived (pixel size).")
                    break

                # Still far: take a short forward step
                ArucoUtils.forward_step(bot, STEP_FWD_M)
                time.sleep(SEARCH_SLEEP_S)
                continue

            time.sleep(LOOP_SLEEP_S)

    except KeyboardInterrupt:
        print("\n[ABORT] Ctrl-C")
    finally:
        try:
            bot.stop()
        except:
            pass
        aru.stop_camera()

if __name__ == "__main__":
    main()
