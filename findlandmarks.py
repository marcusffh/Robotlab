#!/usr/bin/env python3
"""
findlandmarks.py
----------------
SEARCH → ALIGN → APPROACH with distance-aware forward steps.
- Picamera2-only, BGR ArUco detector (matches your working code).
- Stops by *distance* (computed from f_px & 14 cm marker), not pixel-size.
"""

import time
import numpy as np
import cv2

# Import CalibratedRobot; ensure 'exercise1/__init__.py' exists. If not, use sys.path fallback.
try:
    from Exercise1.CalibratedRobot import CalibratedRobot
except ModuleNotFoundError:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "exercise1"))
    from Exercise1.CalibratedRobot import CalibratedRobot

from aruco_utils import ArucoUtils

# ---- Camera / Marker calibration (from your part 1) ----
F_PX      = 1275.0          # focal length in pixels
MARKER_MM = 140.0           # physical tag height (14 cm)
TARGET_ID = None            # set to an int to lock to one ID

# ---- Behavior tuning ----
# Search
SEARCH_STEP_DEG = 12.0      # smaller search pulse to avoid overshoot
SEARCH_SLEEP_S  = 0.08

# Align
PX_TOL              = 36    # larger deadband to avoid ping-pong at distance
PX_KP_DEG_PER_PX    = 0.06  # gentler mapping px->deg to reduce overshoot
MAX_ALIGN_STEP_DEG  = 14.0  # cap micro-turns

# Approach / stopping
STOP_AT_MM          = 450.0 # stand-off distance (mm)
STEP_MIN_M          = 0.10  # min forward step (m)
STEP_MAX_M          = 0.35  # max forward step (m)
STEP_SCALE          = 0.6   # step = clamp((Z-STOP)/1000 * STEP_SCALE, MIN, MAX)

# Robustness
LOST_LIMIT          = 12    # tolerate brief drop-outs
LOOP_SLEEP_S        = 0.04

def estimate_Z_mm(x_px, f_px=F_PX, X_mm=MARKER_MM) -> float:
    # Pinhole: Z = f * X / x
    return (f_px * X_mm) / max(x_px, 1e-6)

def choose_turn_deg(err_px: float) -> float:
    return float(np.clip(err_px * PX_KP_DEG_PER_PX, -MAX_ALIGN_STEP_DEG, MAX_ALIGN_STEP_DEG))

def choose_step_m(Z_mm: float) -> float:
    remaining = max(0.0, Z_mm - STOP_AT_MM) / 1000.0  # -> meters
    step = remaining * STEP_SCALE
    return float(np.clip(step, STEP_MIN_M, STEP_MAX_M))

def main():
    bot = CalibratedRobot()
    aru = ArucoUtils(res=(640, 480), fps=30)  # lower res -> bigger tag in pixels at distance
    aru.start_camera()

    state       = "SEARCH"
    lost        = 0
    need_turn   = True   # in APPROACH: alternate small turn, then a forward step

    print("findlandmarks: SEARCH → ALIGN → APPROACH → DONE")
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
                if det is None:
                    ArucoUtils.rotate_step(bot, SEARCH_STEP_DEG)
                    time.sleep(SEARCH_SLEEP_S)
                    continue
                print(f"[SEARCH→ALIGN] id={det['id']} x={det['x_px']:.1f}px cx={det['cx']:.1f}")
                state = "ALIGN"; lost = 0
                continue

            # common loss handling
            if det is None:
                lost += 1
                if lost >= LOST_LIMIT:
                    print("[LOST] back to SEARCH")
                    state = "SEARCH"; lost = 0
                time.sleep(LOOP_SLEEP_S)
                continue
            lost = 0

            # parse detection
            cx, w = det["cx"], det["w"]
            err_px = cx - (w * 0.5)         # + -> marker right of center
            x_px   = det["x_px"]
            Z_mm   = estimate_Z_mm(x_px)    # distance estimate from your calibration

            # -------- ALIGN --------
            if state == "ALIGN":
                if abs(err_px) <= PX_TOL:
                    print(f"[ALIGN→APPROACH] centered; Z≈{Z_mm:.0f} mm")
                    state = "APPROACH"; need_turn = True
                else:
                    turn_deg = choose_turn_deg(err_px)
                    print(f"[ALIGN] err={err_px:.1f}px -> turn {turn_deg:.1f}°")
                    ArucoUtils.rotate_step(bot, turn_deg)
                    time.sleep(SEARCH_SLEEP_S)
                continue

            # -------- APPROACH --------
            if state == "APPROACH":
                # Stop by distance (more reliable than pixel size)
                if Z_mm <= STOP_AT_MM:
                    bot.stop()
                    print(f"[DONE] close enough: Z≈{Z_mm:.0f} mm (≤ {STOP_AT_MM:.0f} mm)")
                    break

                if need_turn and abs(err_px) > PX_TOL:
                    # small correction turn
                    turn_deg = choose_turn_deg(err_px)
                    print(f"[APPROACH] small turn {turn_deg:.1f}° (err={err_px:.1f}px)")
                    ArucoUtils.rotate_step(bot, turn_deg)
                    need_turn = False
                    time.sleep(SEARCH_SLEEP_S)
                    continue

                # forward step sized by remaining distance
                step_m = choose_step_m(Z_mm)
                print(f"[APPROACH] step {step_m:.2f} m (Z≈{Z_mm:.0f} mm)")
                ArucoUtils.forward_step(bot, step_m)
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
