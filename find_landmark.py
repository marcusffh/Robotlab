#!/usr/bin/env python3
"""
find_landmark_arlo.py
-----------------------------------------
Rotate-search for an ArUco, then drive toward it and stop in front.

Behavior:
  SEARCH : spin in short pulses; after each pulse STOP and check camera
  ALIGN  : micro-rotate to center the marker (pixel deadband)
  APPROACH: short forward steps with heading correction (smoothed)
  DONE   : stop at stand-off distance

Controls (while running):
  - Ctrl-C  -> immediate safe stop
  - type 'q' then Enter -> safe stop (works over SSH)

Requirements:
  - robot.py in same folder (with Robot.go_diff(...) and Robot.stop())
  - OpenCV with aruco module available
  - picamera2 preferred; falls back to GStreamer/OpenCV (headless)
"""

import time, math, os, sys, signal, select
import numpy as np

# =============== CAMERA / MARKER ==================
F_PX        = 1275.0      # calibrated focal length (px) from Part 1
MARKER_MM   = 140.0       # marker physical height (mm) (14 cm)
TARGET_ID   = None        # set to an int (e.g. 6) to lock to a specific ID

# =============== DRIVE / CONTROL ==================
MIN_PWR     = 40          # safe lower bound for go_diff power
MAX_PWR     = 127

# Motor calibration (match your tuned values)
CAL_KL      = 0.98        # left motor scale
CAL_KR      = 1.00        # right motor scale

# Search/align/approach tuning
SEARCH_PWR  = 60          # spin power during SEARCH
TURN_PWR    = 52          # tiny alignment pulses in ALIGN
DRIVE_PWR   = 58          # forward power during APPROACH

SPIN_MS     = 120         # SEARCH spin pulse duration (ms)
ALIGN_MS    = 60          # ALIGN pulse duration (ms)
STEP_MS     = 180         # APPROACH forward burst (ms) — short for stability
PX_TOL      = 28          # deadband in pixels around image center

# Smoother steering:
Kp          = 0.0009      # proportional gain: px -> bias
BASE_BIAS   = -0.06       # constant bias (neg -> slight right) to cancel left drift
EMA_ALPHA   = 0.30        # low-pass filter for pixel error [0..1]

STOP_AT_MM  = 450.0       # stand-off distance to stop in front (mm)
LOST_LIMIT  = 8           # frames lost before returning to SEARCH

# =============== ABORT HANDLING ===================
ABORT = False
def _handle_sig(*_):
    global ABORT
    ABORT = True
signal.signal(signal.SIGINT,  _handle_sig)
signal.signal(signal.SIGTERM, _handle_sig)

def user_requested_quit():
    # type 'q' then Enter to request a clean quit
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        s = sys.stdin.readline().strip().lower()
        if s == 'q':
            return True
    return False

# ================= CAMERA HELPERS =================
def make_camera(width=960, height=720, fps=30):
    """Return (read_fn, release_fn) that yields BGR frames."""
    try:
        from picamera2 import Picamera2
        import cv2
        cam = Picamera2()
        frame_dur = int(1.0/fps * 1_000_000)
        cfg = cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameDurationLimits": (frame_dur, frame_dur)},
            queue=False
        )
        cam.configure(cfg); cam.start(); time.sleep(0.8)
        def read_fn():
            rgb = cam.capture_array("main")
            return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        def release_fn():
            try: cam.stop()
            except: pass
        return read_fn, release_fn
    except Exception as e:
        import cv2
        def gst(w, h, f):
            return ("libcamerasrc ! videobox autocrop=true ! "
                    f"video/x-raw, width=(int){w}, height=(int){h}, framerate=(fraction){f}/1 ! "
                    "videoconvert ! appsink")
        cap = cv2.VideoCapture(gst(width, height, fps), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError(f"Camera init failed: {e}")
        def read_fn():
            ok, frame = cap.read()
            return ok, frame
        def release_fn():
            try: cap.release()
            except: pass
        return read_fn, release_fn

# ================== VISION ========================
def detect_marker(frame_bgr, restrict_id=None):
    """Detect DICT_6X6_250; return dict {id,x_px,cx,w} or None."""
    import cv2
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners_list, ids, _ = aruco.detectMarkers(frame_bgr, dictionary, parameters=parameters)
    if ids is None or len(corners_list) == 0:
        return None
    # choose largest by perimeter (more stable)
    best = None
    for c, mid in zip(corners_list, ids.flatten()):
        if restrict_id is not None and int(mid) != restrict_id:
            continue
        pts = c.reshape(-1, 2)  # TL, TR, BR, BL
        per = (np.linalg.norm(pts[0]-pts[1]) + np.linalg.norm(pts[1]-pts[2]) +
               np.linalg.norm(pts[2]-pts[3]) + np.linalg.norm(pts[3]-pts[0]))
        if best is None or per > best[0]:
            best = (per, int(mid), pts)
    if best is None:
        return None
    _, mid, pts = best
    TL, TR, BR, BL = pts
    v1 = np.linalg.norm(TL - BL)
    v2 = np.linalg.norm(TR - BR)
    x_px = 0.5*(v1+v2)    # vertical pixel height (mean of both sides)
    cx  = float((TL[0]+TR[0]+BR[0]+BL[0]) / 4.0)
    h, w = frame_bgr.shape[:2]
    return {"id": mid, "x_px": float(x_px), "cx": cx, "w": w}

def estimate_Z_mm(x_px, f_px=F_PX, X_mm=MARKER_MM):
    # Z = f * X / x
    return (f_px * X_mm) / max(x_px, 1e-6)

# ================ ROBOT HELPERS ===================
def clamp(p): 
    return 0 if p <= 0 else MAX_PWR if p > MAX_PWR else int(p)

def scaled_lr(power_left, power_right):
    # apply per-side calibration + enforce min power (avoid stalling)
    L = clamp(power_left  * CAL_KL) if power_left  > 0 else 0
    R = clamp(power_right * CAL_KR) if power_right > 0 else 0
    if 0 < L < MIN_PWR: L = MIN_PWR
    if 0 < R < MIN_PWR: R = MIN_PWR
    return L, R

def spin_left(arlo, power, ms):
    L, R = scaled_lr(power, power)
    arlo.go_diff(L, R, 0, 1)  # left backward, right forward
    time.sleep(ms/1000.0); arlo.stop()

def spin_right(arlo, power, ms):
    L, R = scaled_lr(power, power)
    arlo.go_diff(L, R, 1, 0)  # left forward, right backward
    time.sleep(ms/1000.0); arlo.stop()

def align_pulse(arlo, err_px):
    if err_px > 0:  # marker right of center -> spin right
        spin_right(arlo, TURN_PWR, ALIGN_MS)
    else:
        spin_left(arlo, TURN_PWR, ALIGN_MS)

def drive_step_with_bias(arlo, base_power, bias, ms):
    # bias in [-1..1]; positive -> steer right (more power to right wheel)
    Lp = base_power * (1.0 - bias)
    Rp = base_power * (1.0 + bias)
    L, R = scaled_lr(Lp, Rp)
    arlo.go_diff(L, R, 1, 1)
    time.sleep(ms/1000.0); arlo.stop()

# ===================== MAIN =======================
def main():
    import robot as rb
    arlo = rb.Robot()

    read_fn, release_fn = make_camera()
    state = "SEARCH"
    lost = 0
    err_filt = 0.0  # EMA of pixel error

    print("find_landmark_arlo: SEARCH -> ALIGN -> APPROACH -> DONE")
    try:
        while True:
            if ABORT or user_requested_quit():
                print("[ABORT] user requested stop"); break

            ok, frame = read_fn()
            if not ok:
                time.sleep(0.05); continue

            det = detect_marker(frame, restrict_id=TARGET_ID)

            if state == "SEARCH":
                if det is None:
                    spin_left(arlo, SEARCH_PWR, SPIN_MS)  # rotate, stop, check again
                    continue
                state = "ALIGN"; lost = 0
                continue

            if det is None:
                lost += 1
                if lost >= LOST_LIMIT:
                    state = "SEARCH"
                continue

            # Got a detection
            lost = 0
            x_px, cx, w = det["x_px"], det["cx"], det["w"]
            Z = estimate_Z_mm(x_px)
            err = cx - (w/2.0)  # +: marker right of center

            if state == "ALIGN":
                if abs(err) <= PX_TOL:
                    state = "APPROACH"
                else:
                    align_pulse(arlo, err)
                continue

            if state == "APPROACH":
                if Z <= STOP_AT_MM:
                    arlo.stop()
                    print(f"[DONE] stop at Z={Z:.0f} mm")
                    break

                # --- smoothed heading control with deadband + constant bias ---
                err_filt = (1.0 - EMA_ALPHA) * err_filt + EMA_ALPHA * err
                if abs(err_filt) <= PX_TOL:
                    steer = BASE_BIAS
                else:
                    steer = BASE_BIAS + Kp * err_filt
                # clamp
                steer = 0.4 if steer > 0.4 else (-0.4 if steer < -0.4 else steer)

                drive_step_with_bias(arlo, DRIVE_PWR, steer, STEP_MS)
                continue

    except KeyboardInterrupt:
        print("\n[ABORT] Ctrl-C")
    finally:
        try: arlo.stop()
        except: pass
        release_fn()
        print("Exiting.")

if __name__ == "__main__":
    main()
