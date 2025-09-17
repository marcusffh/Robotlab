#!/usr/bin/env python3
"""
find_landmark_arlo.py — slow duty-cycled search + local recover (no 360)
------------------------------------------------------------------------
States:
  SEARCH  : very slow rotation via duty cycle, reading camera every frame
  ALIGN   : gentle micro-rotations to center the marker (deadband)
  APPROACH: short forward steps with smoothed heading correction
  RECOVER : small oscillating turns toward last-seen side until reacquired
  DONE    : stop at stand-off distance

Controls:
  - Ctrl-C          -> safe stop
  - type 'q' + Enter -> safe stop (works over SSH)

Requires:
  - robot.py (Robot.go_diff(...), Robot.stop())
  - OpenCV with aruco, picamera2 preferred (GStreamer fallback)
"""

import time, sys, signal, select
import numpy as np

# ===== Camera / marker =====
F_PX        = 1275.0     # calibrated focal length (px)
MARKER_MM   = 140.0      # marker height (mm)
TARGET_ID   = None       # lock to specific ArUco id (e.g., 6), or None

# ===== Drive / calibration =====
MIN_PWR     = 40
MAX_PWR     = 127
CAL_KL      = 0.98       # tuned scales (left)
CAL_KR      = 1.00       # tuned scales (right)

# ===== Behavior tuning =====
# SEARCH: use duty-cycled spin for truly slow rotation
SEARCH_PWR      = 44     # power while "on" (must be >= MIN_PWR if > 0)
SEARCH_DIR      = +1     # +1 spin right, -1 spin left
SPIN_PERIOD_MS  = 350    # total on+off period (ms)
SPIN_DUTY       = 0.22   # fraction of period spinning (0.22 ≈ 22% ON)

# ALIGN & APPROACH
TURN_PWR        = 42     # min safe power, gentle micro-turns
ALIGN_PULSE_MS  = 90     # slightly longer for smoothness
DRIVE_PWR       = 58     # forward step power
PX_TOL          = 30     # deadband around image center (px)
Kp              = 0.0008 # px -> steering bias (gentle)
BASE_BIAS       = -0.06  # cancels left drift
EMA_ALPHA       = 0.30   # smoothing for pixel error (0..1)

STEP_MS         = 160    # forward step duration (short = stable)
STOP_AT_MM      = 450.0  # stop distance
LOST_LIMIT      = 8      # frames without detection -> lost

# RECOVER (local search, NO 360)
REC_PWR         = 44           # recovery turning power (slow)
REC_PULSE_MS    = 110          # single recovery pulse
REC_MAX_PULSES  = 24           # cap before falling back to SEARCH

# ===== Safe abort =====
ABORT = False
def _sig(*_):  # SIGINT/SIGTERM
    global ABORT; ABORT = True
signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)

def user_requested_quit():
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        s = sys.stdin.readline().strip().lower()
        if s == 'q': return True
    return False

# ===== Camera helpers =====
def make_camera(width=960, height=720, fps=30):
    """Return (read_fn, release_fn) -> (ok, BGR)."""
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
        def gst(w,h,f):
            return ("libcamerasrc ! videobox autocrop=true ! "
                    f"video/x-raw, width=(int){w}, height=(int){h}, framerate=(fraction){f}/1 ! "
                    "videoconvert ! appsink")
        cap = cv2.VideoCapture(gst(width, height, fps), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError(f"Camera init failed: {e}")
        def read_fn():
            ok, frame = cap.read(); return ok, frame
        def release_fn():
            try: cap.release()
            except: pass
        return read_fn, release_fn

# ===== Vision =====
def detect_marker(frame_bgr, restrict_id=None):
    """Detect DICT_6X6_250; return {id,x_px,cx,w} or None."""
    import cv2
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners_list, ids, _ = aruco.detectMarkers(frame_bgr, dictionary, parameters=parameters)
    if ids is None or len(corners_list) == 0:
        return None
    best = None
    for c, mid in zip(corners_list, ids.flatten()):
        if restrict_id is not None and int(mid) != restrict_id:
            continue
        pts = c.reshape(-1,2)  # TL,TR,BR,BL
        per = (np.linalg.norm(pts[0]-pts[1]) + np.linalg.norm(pts[1]-pts[2]) +
               np.linalg.norm(pts[2]-pts[3]) + np.linalg.norm(pts[3]-pts[0]))
        if best is None or per > best[0]:
            best = (per, int(mid), pts)
    if best is None: return None
    _, mid, pts = best
    TL, TR, BR, BL = pts
    v1 = np.linalg.norm(TL - BL)
    v2 = np.linalg.norm(TR - BR)
    x_px = 0.5*(v1+v2)
    cx  = float((TL[0]+TR[0]+BR[0]+BL[0]) / 4.0)
    h, w = frame_bgr.shape[:2]
    return {"id": mid, "x_px": float(x_px), "cx": cx, "w": w}

def estimate_Z_mm(x_px, f_px=F_PX, X_mm=MARKER_MM):
    return (f_px * X_mm) / max(x_px, 1e-6)

# ===== Robot helpers =====
def clamp(p): 
    return 0 if p<=0 else MAX_PWR if p>MAX_PWR else int(p)

def scaled_lr(pl, pr):
    L = clamp(pl * CAL_KL) if pl>0 else 0
    R = clamp(pr * CAL_KR) if pr>0 else 0
    if 0 < L < MIN_PWR: L = MIN_PWR
    if 0 < R < MIN_PWR: R = MIN_PWR
    return L, R

def spin_continuous(arlo, power, dir_sign):
    power = max(MIN_PWR, min(MAX_PWR, int(power)))
    L,R = scaled_lr(power, power)
    if dir_sign >= 0: arlo.go_diff(L, R, 1, 0)  # right spin
    else:             arlo.go_diff(L, R, 0, 1)  # left spin

def stop(arlo): arlo.stop()

def spin_pulse(arlo, power, dir_sign, ms):
    spin_continuous(arlo, power, dir_sign)
    time.sleep(ms/1000.0); arlo.stop()

def drive_step_with_bias(arlo, base_power, bias, ms):
    bias = max(min(bias, 0.5), -0.5)
    Lp = base_power * (1.0 - bias)
    Rp = base_power * (1.0 + bias)
    L,R = scaled_lr(Lp, Rp)
    arlo.go_diff(L, R, 1, 1)
    time.sleep(ms/1000.0); arlo.stop()

# ===== Duty-cycled slow spin =====
_spin_state = {"on": False, "t0": 0.0}
def spin_pwm_step(arlo, power, dir_sign, period_ms=SPIN_PERIOD_MS, duty=SPIN_DUTY):
    """
    Toggle between spinning and stopped within each period so the average
    angular speed is much slower, while still reading the camera every loop.
    """
    now = time.time()
    if _spin_state["t0"] == 0.0:
        _spin_state["t0"] = now
        _spin_state["on"] = False

    elapsed = (now - _spin_state["t0"]) * 1000.0
    on_ms   = duty * period_ms
    off_ms  = (1.0 - duty) * period_ms

    if _spin_state["on"]:
        if elapsed >= on_ms:
            stop(arlo)
            _spin_state["on"] = False
            _spin_state["t0"] = now
    else:
        if elapsed >= off_ms:
            spin_continuous(arlo, power, dir_sign)
            _spin_state["on"] = True
            _spin_state["t0"] = now

# ===== Main =====
def main():
    import robot as rb
    arlo = rb.Robot()
    read_fn, release_fn = make_camera()

    state = "SEARCH"
    lost = 0
    err_filt = 0.0
    last_err_sign = +1   # +1 = last seen to the right; -1 = left
    rec_pulses = 0

    print("SM: SEARCH (duty) -> ALIGN -> APPROACH -> RECOVER(no 360) -> DONE")
    try:
        while True:
            if ABORT or user_requested_quit():
                print("[ABORT] user stop"); break

            ok, frame = read_fn()
            if not ok:
                time.sleep(0.01); continue

            # ===== SEARCH: very slow via duty-cycle, read camera every frame =====
            if state == "SEARCH":
                spin_pwm_step(arlo, SEARCH_PWR, SEARCH_DIR)  # << slow spin
                det = detect_marker(frame, restrict_id=TARGET_ID)
                if det is None:
                    continue
                stop(arlo)
                state = "ALIGN"
                lost = 0
                # fall through with det
            else:
                det = detect_marker(frame, restrict_id=TARGET_ID)

            # ===== detection / loss handling =====
            if det is None:
                lost += 1
                if state in ("ALIGN","APPROACH"):
                    if lost >= LOST_LIMIT:
                        state = "RECOVER"
                        rec_pulses = 0
                        stop(arlo)
                        continue
                    continue
                elif state == "RECOVER":
                    pass
                else:
                    continue
            else:
                x_px, cx, w = det["x_px"], det["cx"], det["w"]
                Z = estimate_Z_mm(x_px)
                err = cx - (w/2.0)
                last_err_sign = +1 if err > 0 else -1
                lost = 0

            # ===== State logic =====
            if state == "ALIGN":
                if abs(err) <= PX_TOL:
                    state = "APPROACH"
                    continue
                spin_pulse(arlo, TURN_PWR, +1 if err>0 else -1, ALIGN_PULSE_MS)
                continue

            if state == "APPROACH":
                if Z <= STOP_AT_MM:
                    stop(arlo)
                    print(f"[DONE] stop at Z={Z:.0f} mm")
                    break
                # smoothed heading
                err_filt = (1.0-EMA_ALPHA)*err_filt + EMA_ALPHA*err
                steer = BASE_BIAS if abs(err_filt) <= PX_TOL else (BASE_BIAS + Kp*err_filt)
                steer = 0.4 if steer>0.4 else (-0.4 if steer<-0.4 else steer)
                drive_step_with_bias(arlo, DRIVE_PWR, steer, STEP_MS)
                continue

            if state == "RECOVER":
                # small oscillating local search around last seen side (no 360)
                dir_sign = last_err_sign if (rec_pulses % 2 == 0) else -last_err_sign
                spin_pulse(arlo, REC_PWR, dir_sign, REC_PULSE_MS)

                ok2, frame2 = read_fn()
                det2 = detect_marker(frame2, restrict_id=TARGET_ID) if ok2 else None

                if det2 is not None:
                    x_px, cx, w = det2["x_px"], det2["cx"], det2["w"]
                    err = cx - (w/2.0)
                    last_err_sign = +1 if err>0 else -1
                    stop(arlo)
                    state = "ALIGN"
                    lost = 0
                    continue

                rec_pulses += 1
                if rec_pulses >= REC_MAX_PULSES:
                    stop(arlo)
                    state = "SEARCH"
                    continue
                continue

    except KeyboardInterrupt:
        print("\n[ABORT] Ctrl-C")
    finally:
        try: stop(arlo)
        except: pass
        release_fn()
        print("Exiting.")

if __name__ == "__main__":
    main()
