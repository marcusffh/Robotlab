#!/usr/bin/env python3
"""
findlandmarks.py
----------------
Rotate to find an ArUco marker, ALIGN to center (pixel-only), and APPROACH in steps.
Uses your CalibratedRobot (turn_angle, drive_distance, stop) + the detector that
already works for you (ArucoUtils.detect_one).
"""

import time
import numpy as np
import cv2

# Import CalibratedRobot; ensure 'exercise1/__init__.py' exists and run from project root
try:
    from exercise1.CalibratedRobot import CalibratedRobot
except ModuleNotFoundError:
    # fallback if you prefer running this file from inside Robotlab/Robotlab/
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "exercise1"))
    from CalibratedRobot import CalibratedRobot

from aruco_utils import ArucoUtils

# ---- Marker & detection (pixel-only) ----
TARGET_ID = None  # e.g., set to 6 to target a specific ID only

# ---- Minimal tuning (match your working behavior) ----
SEARCH_STEP_DEG       = 15.0   # rotate a bit, stop, check
SEARCH_SLEEP_S        = 0.10

PX_TOL                = 28     # deadband around image center (pixels)
PX_KP_DEG_PER_PX      = 0.08   # map pixel error -> small turn (deg)
MAX_ALIGN_STEP_DEG    = 18.0

STEP_FWD_M            = 0.25   # forward step length
STOP_SIDE_PX          = 240    # stop when tag looks this big (proxy for distance)

LOST_LIMIT            = 8
LOOP_SLEEP_S          = 0.05

def main():
    bot = CalibratedRobot()
    aru = ArucoUtils(res=(960, 720), fps=30)  # same cam settings as your working script
    aru.start_camera()

    state = "SEARCH"
    lost  = 0

    print("findlandmarks: SEARCH → ALIGN → APPROACH → DONE")
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
                print(f"[SEARCH→ALIGN] id={det['id']} x={det['x_px']:.1f}px cx={det['cx']:.1f}")
                state = "ALIGN"
                lost  = 0
                continue

            if det is None:
                lost += 1
                if lost >= LOST_LIMIT:
                    print("[LOST] back to SEARCH")
                    state = "SEARCH"
                    lost = 0
                time.sleep(LOOP_SLEEP_S)
                continue

            # We have a detection now
            cx, w = det["cx"], det["w"]
            x_px  = det["x_px"]
            err_px = cx - (w * 0.5)

            if state == "ALIGN":
                if abs(err_px) <= PX_TOL:
                    print("[ALIGN→APPROACH] centered")
                    state = "APPROACH"
                else:
                    step_deg = float(np.clip(err_px * PX_KP_DEG_PER_PX, -MAX_ALIGN_STEP_DEG, MAX_ALIGN_STEP_DEG))
                    print(f"[ALIGN] err={err_px:.1f}px -> turn {step_deg:.1f}°")
                    ArucoUtils.rotate_step(bot, step_deg)
                    time.sleep(SEARCH_SLEEP_S)
                continue

            if state == "APPROACH":
                if x_px >= STOP_SIDE_PX:
                    bot.stop()
                    print("[DONE] close enough (pixel size threshold)")
                    break
                print(f"[APPROACH] step forward; size={x_px:.1f}px (target {STOP_SIDE_PX}px)")
                ArucoUtils.forward_step(bot, STEP_FWD_M)
                time.sleep(SEARCH_SLEEP_S)
                continue

            time.sleep(LOOP_SLEEP_S)

    except KeyboardInterrupt:
        print("\n[ABORT] Ctrl-C")
    finally:
        try: bot.stop()
        except: pass
        aru.stop_camera()

if __name__ == "__main__":
    main()
