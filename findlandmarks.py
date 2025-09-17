#!/usr/bin/env python3
"""
find_landmarks.py
-----------------
Task 2: Rotate to find an ArUco landmark, align toward it, and drive to it.
- Uses your CalibratedRobot API (turn_angle, drive_distance, drive, stop).
- Uses ArucoUtils for camera + detection (+ optional pose).

Simple state machine:
  SEARCH  -> rotate stepwise until a marker is seen
  APPROACH-> (pose) yaw-align in small turns, then step forward; repeat
           (no pose) pixel-align in small turns, then step forward; repeat
  LOST    -> if detection disappears during approach, go back to SEARCH

Keep it minimal and robust. Tune the constants below on your robot.
"""

import time
import numpy as np
import cv2

from Exercise1.CalibratedRobot import CalibratedRobot
from arucoutils import ArucoUtils, CameraConfig, Intrinsics

# ------------- Tuning -------------
# Search behaviour
SEARCH_STEP_DEG = 25.0          # rotate this much per search step
SEARCH_SLEEP_S  = 0.1           # small settle time after each turn

# Alignment (pose-based)
YAW_TOL_RAD     = 0.06          # ~3.4 deg tolerance
YAW_KP_DEG_PER_RAD = 35.0       # map yaw(rad) -> turn step (deg)
MAX_ALIGN_STEP_DEG = 18.0       # cap per-step rotation

# Alignment (pixel-based, if no pose)
PX_TOL          = 12            # deadband in pixels
PX_KP_DEG_PER_PX = 0.08         # map pixel error -> turn step (deg)
MAX_ALIGN_STEP_DEG_PX = 18.0

# Forward motion
STEP_FWD_M      = 0.25          # forward distance per step when still far
STEP_FWD_MIN_M  = 0.10          # minimum step when very close
STOP_DIST_M     = 0.22          # stop at this Z (if pose available)
STOP_SIDE_PX    = 240           # stop when the marker looks this big (no pose)

# Safety / loop
LOST_BACK_TO_SEARCH = 6         # frames to tolerate lost detections while approaching
LOOP_SLEEP_S        = 0.05      # main loop pacing
RESTRICT_IDS        = None      # e.g. [7] if you only want a specific marker
# ---------------------------------


def main():
    # If you have calibration, fill these and set marker_size_m (meters).
    intr = None   # Example: Intrinsics(fx, fy, cx, cy, dist=np.zeros(5))
    marker_size_m = None  # e.g., 0.14 for 14 cm

    bot = CalibratedRobot()
    aru = ArucoUtils(cam_cfg=CameraConfig(), intrinsics=intr, marker_size_m=marker_size_m)
    aru.start_camera()

    state = "SEARCH"
    lost_counter = 0

    try:
        while True:
            ok, frame = aru.read()
            if not ok:
                bot.stop()
                print("Camera read failed.")
                break

            h, w = frame.shape[:2]
            dets = aru.detect(frame, restrict_ids=RESTRICT_IDS)
            det = ArucoUtils.choose_largest(dets)

            if state == "SEARCH":
                if det is None:
                    # rotate a bit and try again
                    ArucoUtils.rotate_step(bot, SEARCH_STEP_DEG)
                    time.sleep(SEARCH_SLEEP_S)
                else:
                    state = "APPROACH"
                    lost_counter = LOST_BACK_TO_SEARCH
                    continue  # start approach immediately

            elif state == "APPROACH":
                if det is None:
                    # temporary loss while moving -> count down, then back to search
                    if lost_counter > 0:
                        lost_counter -= 1
                        time.sleep(LOOP_SLEEP_S)
                        continue
                    else:
                        state = "SEARCH"
                        continue

                # We have a detection -> align then step forward
                if det.tvec is not None:
                    # --- Pose-based path ---
                    yaw = ArucoUtils.yaw_from_tvec(det.tvec)
                    z = float(det.tvec[2])

                    if abs(yaw) > YAW_TOL_RAD:
                        step_deg = float(np.clip(yaw * YAW_KP_DEG_PER_RAD, -MAX_ALIGN_STEP_DEG, MAX_ALIGN_STEP_DEG))
                        ArucoUtils.rotate_step(bot, step_deg)
                        time.sleep(SEARCH_SLEEP_S)
                        continue

                    # facing target; step forward toward stop distance
                    if z > STOP_DIST_M:
                        remaining = max(0.0, z - STOP_DIST_M)
                        step = float(np.clip(remaining, STEP_FWD_MIN_M, STEP_FWD_M))
                        ArucoUtils.go_forward_step(bot, step)
                        time.sleep(SEARCH_SLEEP_S)
                        continue
                    else:
                        bot.stop()
                        print("Arrived (pose).")
                        break

                else:
                    # --- Pixel-based path (no intrinsics) ---
                    err_px = det.center_xy[0] - (w * 0.5)
                    if abs(err_px) > PX_TOL:
                        step_deg = float(np.clip(err_px * PX_KP_DEG_PER_PX, -MAX_ALIGN_STEP_DEG_PX, MAX_ALIGN_STEP_DEG_PX))
                        ArucoUtils.rotate_step(bot, step_deg)
                        time.sleep(SEARCH_SLEEP_S)
                        continue

                    # centered; step forward until the marker fills enough pixels
                    if det.side_px < STOP_SIDE_PX:
                        ArucoUtils.go_forward_step(bot, STEP_FWD_M)
                        time.sleep(SEARCH_SLEEP_S)
                        continue
                    else:
                        bot.stop()
                        print("Arrived (pixel size).")
                        break

            # tiny pace to avoid busy-waiting
            time.sleep(LOOP_SLEEP_S)

    finally:
        bot.stop()
        aru.stop_camera()


if __name__ == "__main__":
    main()
