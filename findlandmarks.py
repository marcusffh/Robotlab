#!/usr/bin/env python3
"""
findlandmarks.py
----------------
SEARCH → ALIGN → APPROACH with distance-aware steps and verbose prints.
- Picamera2-only, BGR ArUco detector (DICT_6X6_250).
- Stops by *distance* using f=1275 px, marker=140 mm.
"""

import time
import numpy as np
import cv2

# Import CalibratedRobot; ensure 'exercise1/__init__.py' exists. If not, fallback to sys.path tweak.
try:
    from Exercise1.CalibratedRobot import CalibratedRobot
except ModuleNotFoundError:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "exercise1"))
    from Exercise1.CalibratedRobot import CalibratedRobot

from aruco_utils import ArucoUtils

# ---- Camera / Marker calibration (from your Part 1) ----
F_PX      = 1275.0          # focal length in pixels
MARKER_MM = 140.0           # 14 cm marker
TARGET_ID = 6            # set to an int to lock to one ID

# ---- Behavior tuning ----
# Search
SEARCH_STEP_DEG = 10.0      # smaller pulse → finer sweep
SEARCH_SLEEP_S  = 0.08

# Align
PX_TOL              = 36    # larger deadband to avoid ping-pong when far
PX_KP_DEG_PER_PX    = 0.05  # gentle mapping px->deg
MAX_ALIGN_STEP_DEG  = 12.0  # cap corrections

# Approach / stopping
STOP_AT_MM          = 450.0 # stand-off distance (mm)
STEP_MIN_M          = 0.08  # min forward step (m)
STEP_MAX_M          = 0.35  # max forward step (m)
STEP_SCALE          = 0.6   # step = clamp((Z-STOP)/1000 * SCALE, MIN, MAX)

# Robustness
LOST_LIMIT          = 14    # tolerate brief drop-outs
LOOP_SLEEP_S        = 0.04

def estimate_Z_mm(x_px, f_px=F_PX, X_mm=MARKER_MM) -> float:
    return (f_px * X_mm) / max(x_px, 1e-6)

def choose_turn_deg(err_px: float) -> float:
    return float(np.clip(err_px * PX_KP_DEG_PER_PX, -MAX_ALIGN_STEP_DEG, MAX_ALIGN_STEP_DEG))

def choose_step_m(Z_mm: float) -> float:
    remaining = max(0.0, Z_mm - STOP_AT_MM) / 1000.0  # -> meters
    step = remaining * STEP_SCALE
    return float(np.clip(step, STEP_MIN_M, STEP_MAX_M))

def main():
    bot = CalibratedRobot()
    aru = ArucoUtils(res=(1640, 1232), fps=30)  # high-res for far detection
    aru.start_camera()

    state       = "SEARCH"
    lost        = 0
    need_turn   = True   # in APPROACH: alternate small turn → forward step

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
                print("[SEARCH] rotating {:.1f}° and checking camera…".format(SEARCH_STEP_DEG))
                if det is None:
                    ArucoUtils.rotate_step(bot, SEARCH_STEP_DEG)
                    time.sleep(SEARCH_SLEEP_S)
                    continue
                x_px, cx, w = det["x_px"], det["cx"], det["w"]
                err_px = cx - (w * 0.5)
                Z_mm   = estimate_Z_mm(x_px)
                print(f"[FOUND] id={det['id']} at ~{Z_mm:.0f} mm | size={x_px:.1f}px | err={err_px:.1f}px")
                state = "ALIGN"; lost = 0
                continue

            # common loss handling
            if det is None:
                lost += 1
                print(f"[LOST] frame without detection ({lost}/{LOST_LIMIT})")
                if lost >= LOST_LIMIT:
                    print("[LOST] returning to SEARCH")
                    state = "SEARCH"; lost = 0
                time.sleep(LOOP_SLEEP_S)
                continue
            lost = 0

            # parse detection
            cx, w = det["cx"], det["w"]
            err_px = cx - (w * 0.5)         # + -> marker right of center
            x_px   = det["x_px"]
            Z_mm   = estimate_Z_mm(x_px)

            # -------- ALIGN --------
            if state == "ALIGN":
                if abs(err_px) <= PX_TOL:
                    print(f"[ALIGN→APPROACH] centered; Z≈{Z_mm:.0f} mm")
                    state = "APPROACH"; need_turn = True
                else:
                    turn_deg = choose_turn_deg(err_px)
                    print(f"[ALIGN] err={err_px:.1f}px → turn {turn_deg:.1f}°")
                    ArucoUtils.rotate_step(bot, turn_deg)
                    time.sleep(SEARCH_SLEEP_S)
                continue

            # -------- APPROACH --------
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
