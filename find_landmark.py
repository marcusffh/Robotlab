#!/usr/bin/env python3
"""
find_landmark_arlo.py — Rotate-search for an ArUco, then approach and stop.

Behavior:
  1) Rotate in short pulses; after each pulse, stop and check camera.
  2) If no marker, keep rotating.
  3) If marker found, align + drive toward it in short steps with heading correction.
  4) Stop when within STOP_AT_MM.

Tuned motor usage:
  - Uses your robot.py go_diff style with safe power [40..127]
  - Applies left/right calibration scalars (CAL_KL, CAL_KR) like in your code

Headless (no GUI). Works over SSH.
"""

import time, math, os
import numpy as np

# ====== CAMERA / MARKER PARAMS (edit if needed) ======
F_PX        = 1275.0   # your calibrated focal length in pixels (from Part 1)
MARKER_MM   = 140.0    # your marker's physical height in mm (14 cm)
TARGET_ID   = None     # set to an int to restrict to a specific ArUco ID (e.g., 6)
# ====== ROBOT / CONTROL PARAMS (tuned pulses) ========
MIN_PWR     = 40
MAX_PWR     = 127
CAL_KL      = 0.98     # your left motor scale (from your calibration code)
CAL_KR      = 1.00     # your right motor scale
SEARCH_PWR  = 60       # spin power while searching
TURN_PWR    = 55       # in-place alignment power
DRIVE_PWR   = 60       # forward power while approaching
SPIN_MS     = 120      # spin pulse (ms) during SEARCH
ALIGN_MS    = 60       # tiny alignment pulse (ms) in ALIGN
STEP_MS     = 350      # forward step (ms) in APPROACH
PX_TOL      = 20       # acceptable pixel offset from image center
STOP_AT_MM  = 450.0    # stop when estimated distance <= this
LOST_LIMIT  = 8        # frames lost before returning to SEARCH
# =====================================================

# ---------- camera helpers ----------
def make_camera(width=960, height=720, fps=30):
    """Return (read_fn, release_fn) -> (ok, BGR). Prefers picamera2."""
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

# ---------- vision ----------
def detect_marker(frame_bgr, restrict_id=None):
    """Detect DICT_6X6_250; return dict with x_px (vertical), center (cx), img width w, id; else None."""
    import cv2
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners_list, ids, _ = aruco.detectMarkers(frame_bgr, dictionary, parameters=parameters)
    if ids is None or len(corners_list)==0:
        return None
    # choose largest by perimeter for stability
    best = None
    for c, mid in zip(corners_list, ids.flatten()):
        if restrict_id is not None and int(mid)!=restrict_id: 
            continue
        pts = c.reshape(-1,2)  # TL,TR,BR,BL
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
    x_px = 0.5*(v1+v2)
    cx  = float((TL[0]+TR[0]+BR[0]+BL[0]) / 4.0)
    h, w = frame_bgr.shape[:2]
    return {"id": mid, "x_px": float(x_px), "cx": cx, "w": w}

def estimate_Z_mm(x_px, f_px=F_PX, X_mm=MARKER_MM):
    # Z = f * X / x
    return (f_px * X_mm) / max(x_px, 1e-6)

# ---------- robot helpers (uses your robot.py + calibration) ----------
def clamp(p): 
    return 0 if p<=0 else MAX_PWR if p>MAX_PWR else int(p)

def scaled_lr(power_left, power_right):
    # apply your per-side scaling, respect min-power rule
    L = clamp(power_left  * CAL_KL) if power_left  > 0 else 0
    R = clamp(power_right * CAL_KR) if power_right > 0 else 0
    if 0 < L < MIN_PWR: L = MIN_PWR
    if 0 < R < MIN_PWR: R = MIN_PWR
    return L, R

def spin_left(arlo, power, ms):
    L,R = scaled_lr(power, power)
    arlo.go_diff(L, R, 0, 1)  # left backward, right forward
    time.sleep(ms/1000.0); arlo.stop()

def spin_right(arlo, power, ms):
    L,R = scaled_lr(power, power)
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
    L,R = scaled_lr(Lp, Rp)
    arlo.go_diff(L, R, 1, 1)
    time.sleep(ms/1000.0); arlo.stop()

# ---------- main ----------
def main():
    import robot as rb
    arlo = rb.Robot()
    read_fn, release_fn = make_camera()

    STATE = "SEARCH"
    lost  = 0
    print("find_landmark: SEARCH -> ALIGN -> APPROACH -> DONE")
    try:
        while True:
            ok, frame = read_fn()
            if not ok:
                time.sleep(0.05); continue

            det = detect_marker(frame, restrict_id=TARGET_ID)

            if STATE == "SEARCH":
                if det is None:
                    spin_left(arlo, SEARCH_PWR, SPIN_MS)   # rotate, then stop & check again
                    continue
                STATE = "ALIGN"; lost = 0
                continue

            if det is None:
                lost += 1
                if lost >= LOST_LIMIT:
                    STATE = "SEARCH"
                continue

            # we have a detection
            lost = 0
            x_px, cx, w = det["x_px"], det["cx"], det["w"]
            Z = estimate_Z_mm(x_px)
            err = cx - (w/2.0)

            if STATE == "ALIGN":
                if abs(err) <= PX_TOL:
                    STATE = "APPROACH"
                else:
                    align_pulse(arlo, err)
                continue

            if STATE == "APPROACH":
                if Z <= STOP_AT_MM:
                    arlo.stop()
                    print(f"[DONE] stop at Z={Z:.0f} mm")
                    break
                # proportional steering based on pixel error
                Kp = 0.0015                      # px -> bias gain (tune if needed)
                bias = max(min(Kp * err, 0.5), -0.5)
                drive_step_with_bias(arlo, DRIVE_PWR, bias, STEP_MS)
                continue

    except KeyboardInterrupt:
        print("\n[ABORT] Ctrl+C")
    finally:
        try: arlo.stop()
        except: pass
        release_fn()
        print("Exiting.")

if __name__ == "__main__":
    main()
