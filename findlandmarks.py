#!/usr/bin/env python3
"""
find_landmarks.py
-----------------
Locate an ArUco, align, and drive straight toward it using ONLY CalibratedRobot
(turn_angle, drive_distance, stop). No raw wheel power -> no right-veer.

Behavior:
  SEARCH  : small rotate pulse, STOP, read camera (no blur).
  ALIGN   : micro turns (deadband) using sign-correct mapping.
  APPROACH: alternate (tiny correction -> straight forward step).
  DONE    : stop when Z (from x_px) <= STOP_AT_MM.

Prints clearly while searching, when found, and while moving.
"""

import time
import numpy as np
import cv2

# ---- Bring in your robot class ----
try:
    from Exercise1.CalibratedRobot import CalibratedRobot
except ModuleNotFoundError:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "exercise1"))
    from Exercise1.CalibratedRobot import CalibratedRobot

from aruco_utils import ArucoUtils

# ---- Camera / marker calibration ----
F_PX      = 1275.0          # focal length (px)
MARKER_MM = 140.0           # 14 cm tag
TARGET_ID = None            # set to an int to lock to a specific id

# ---- Search / align tuning ----
SEARCH_STEP_DEG = 8.0       # small rotate pulse (stop-to-look)
SEARCH_SLEEP_S  = 0.06

PX_TOL              = 30    # deadband around center (pixels)
PX_KP_DEG_PER_PX    = 0.05  # << SIGN MATTERS: positive err (marker RIGHT) => need RIGHT turn (NEG degrees)
MAX_ALIGN_STEP_DEG  = 10.0

# ---- Forward-step policy (distance-aware) ----
STOP_AT_MM          = 450.0 # stand-off distance
STEP_MIN_M          = 0.08
STEP_MAX_M          = 0.35
STEP_SCALE          = 0.6   # step = clamp((Z-STOP)/1000 * SCALE, MIN, MAX)

# ---- Robustness ----
LOST_LIMIT          = 12
LOOP_SLEEP_S        = 0.04

def estimate_Z_mm(x_px, f_px=F_PX, X_mm=MARKER_MM) -> float:
    # Pinhole: Z = f * X / x
    return (f_px * X_mm) / max(x_px, 1e-6)

def choose_turn_deg(err_px: float) -> float:
    """
    err_px = cx - (w/2). Positive -> marker is to the RIGHT of center.
    To center, we must turn RIGHT => NEGATIVE angle for CalibratedRobot.turn_angle.
    """
    step = -(err_px * PX_KP_DEG_PER_PX)
    return float(np.clip(step, -MAX_ALIGN_STEP_DEG, MAX_ALIGN_STEP_DEG))

def choose_step_m(Z_mm: float) -> float:
    remaining = max(0.0, Z_mm - STOP_AT_MM) / 1000.0  # meters
    step = remaining * STEP_SCALE
    return float(np.clip(step, STEP_MIN_M, STEP_MAX_M))

def main():
    bot = CalibratedRobot()

    # Camera matches what worked for you (Picamera2 BGR, default params)
    aru = ArucoUtils(res=(960, 720), fps=30)
    aru.start_camera()

    state     = "SEARCH"
    lost      = 0
    need_turn = True   # in APPROACH: alternate tiny turn -> forward step

    print("find_landmarks: SEARCH → ALIGN → APPROACH → DONE (no raw wheel control)")
    try:
        while True:
            ok, frame = aru.read()
            if not ok:
                bot.stop()
                time.sleep(0.05)
                continue

            det = aru.detect_one(frame, restrict_id=TARGET_ID)

            # -------- SEARCH --------
            if state == "SEARCH":
                print(f"[SEARCH] rotate {SEARCH_STEP_DEG:.1f}° then check…")
                if det is None:
                    ArucoUtils.rotate_step(bot, SEARCH_STEP_DEG)  # small left pulse
                    time.sleep(SEARCH_SLEEP_S)
                    continue

                # Found
                x_px, cx, w = det["x_px"], det["cx"], det["w"]
                err_px = cx - (w * 0.5)
                Z_mm   = estimate_Z_mm(x_px)
                print(f"[FOUND] id={det['id']}  Z≈{Z_mm:.0f} mm  size={x_px:.1f}px  err={err_px:.1f}px")
                state = "ALIGN"; lost = 0
                continue

            # -------- LOSS HANDLING --------
            if det is None:
                lost += 1
                print(f"[LOST] no detection ({lost}/{LOST_LIMIT})")
                if lost >= LOST_LIMIT:
                    print("[LOST] returning to SEARCH")
                    state = "SEARCH"; lost = 0
                time.sleep(LOOP_SLEEP_S)
                continue
            lost = 0

            # Parse detection
            cx, w = det["cx"], det["w"]
            err_px = cx - (w * 0.5)    # +: marker right of center
            x_px   = det["x_px"]
            Z_mm   = estimate_Z_mm(x_px)

            # -------- ALIGN --------
            if state == "ALIGN":
                if abs(err_px) <= PX_TOL:
                    print(f"[ALIGN→APPROACH] centered; Z≈{Z_mm:.0f} mm")
                    state = "APPROACH"; need_turn = True
                else:
                    turn_deg = choose_turn_deg(err_px)  # NEG when marker is right -> turn right
                    print(f"[ALIGN] err={err_px:.1f}px → turn {turn_deg:.1f}°")
                    ArucoUtils.rotate_step(bot, turn_deg)
                    time.sleep(SEARCH_SLEEP_S)
                continue

            # -------- APPROACH (no raw motor bias; straight-step via CalibratedRobot) --------
            if state == "APPROACH":
                if Z_mm <= STOP_AT_MM:
                    bot.stop()
                    print(f"[DONE] close enough: Z≈{Z_mm:.0f} mm (≤ {STOP_AT_MM:.0f} mm)")
                    break

                if need_turn and abs(err_px) > PX_TOL:
                    turn_deg = choose_turn_deg(err_px)
                    print(f"[MOVE] correcting heading: turn {turn_deg:.1f}° (err={err_px:.1f}px, Z≈{Z_mm:.0f} mm)")
                    ArucoUtils.rotate_step(bot, turn_deg)
                    need_turn = False
                    time.sleep(SEARCH_SLEEP_S)
                    continue

                step_m = choose_step_m(Z_mm)
                print(f"[MOVE] forward {step_m:.2f} m (Z≈{Z_mm:.0f} mm)")
                ArucoUtils.forward_step(bot, step_m)  # straight, balanced via your CalibratedRobot
                need_turn = True
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
