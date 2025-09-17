#!/usr/bin/env python3
"""
find_landmark_arlo.py — continuous search + local recover (no 360)
------------------------------------------------------------------
States:
  SEARCH  : slow continuous rotation, read camera every frame
  ALIGN   : micro-rotate to center marker (pixel deadband)
  APPROACH: short forward steps with smoothed heading correction
  RECOVER : small oscillating turns toward last-seen side until reacquired
  DONE    : stop at standoff

Controls:
  Ctrl-C          -> safe stop
  type 'q'+Enter  -> safe stop (over SSH)

Requires:
  - robot.py (Robot.go_diff(...), Robot.stop())
  - OpenCV (aruco), picamera2 preferred, GStreamer fallback
"""

import time, sys, signal, select
import numpy as np

# ===== Camera / marker =====
F_PX        = 1275.0     # calibrated focal length (px)
MARKER_MM   = 140.0      # marker height (mm)
TARGET_ID   = None       # lock to specific ArUco id (e.g. 6), or None

# ===== Drive / calibration =====
MIN_PWR     = 40
MAX_PWR     = 127
CAL_KL      = 0.98       # your tuned scales
CAL_KR      = 1.00

# ===== Behavior tuning =====
SEARCH_PWR      = 44     # slow rotation power
SEARCH_DIR      = +1     # +1 spin right, -1 spin left

TURN_PWR        = 50     # align pulse power
DRIVE_PWR       = 58     # forward step power
PX_TOL          = 28     # deadband around image center (px)
Kp              = 0.0009 # px -> steering bias
BASE_BIAS       = -0.06  # cancels left drift
EMA_ALPHA       = 0.30   # smoothing for pixel error

STEP_MS         = 180    # forward step duration
STOP_AT_MM      = 450.0  # stop distance
LOST_LIMIT      = 8      # frames without detection -> lost

# RECOVER (local search, no 360)
REC_PWR         = 46           # recovery turning power
REC_PULSE_MS    = 120          # a single recovery pulse
REC_MAX_PULSES  = 20           # max pulses before giving up to SEARCH
REC_BIAS_INC    = 1            # expand pattern by flipping direction after each pulse

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
    if dir_sign >= 0: arlo.go_diff(L, R, 1, 0)
    else:             arlo.go_diff(L, R, 0, 1)

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

def stop(arlo): arlo.stop()

# ===== Main =====
def main():
    import robot as rb
    arlo = rb.Robot()
    read_fn, release_fn = make_camera()

    state = "SEARCH"
    lost = 0
    err_filt = 0.0
    last_err_sign = +1   # +1 means marker was last seen on right; -1 on left
    rec_pulses = 0       # pulses used in RECOVER (to flip small directions)

    print("SM: SEARCH -> ALIGN -> APPROACH -> (RECOVER on loss) -> DONE")
    try:
        while True:
            if ABORT or user_requested_quit():
                print("[ABORT] user stop"); break

            ok, frame = read_fn()
            if not ok:
                time.sleep(0.01); continue

            # ===== SEARCH: continuous slow rotation, read each frame =====
            if state == "SEARCH":
                spin_continuous(arlo, SEARCH_PWR, SEARCH_DIR)
                det = detect_marker(frame, restrict_id=TARGET_ID)
                if det is None:
                    continue
                stop(arlo)
                state = "ALIGN"
                lost = 0
                # fall through with det available
            else:
                det = detect_marker(frame, restrict_id=TARGET_ID)

            # ===== handle detection / loss =====
            if det is None:
                lost += 1
                if state in ("ALIGN","APPROACH"):
                    if lost >= LOST_LIMIT:
                        # enter local RECOVER using last_err_sign (no 360)
                        state = "RECOVER"
                        rec_pulses = 0
                        stop(arlo)
                        continue
                    # keep looping; maybe next frame sees it again
                    continue
                elif state == "RECOVER":
                    # if we’re already recovering, we execute below
                    pass
                else:
                    # e.g., still in SEARCH without detection -> loop
                    continue
            else:
                # update last side where we saw the marker
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
                # micro pulses toward error sign
                spin_pulse(arlo, TURN_PWR, +1 if err>0 else -1, 60)
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
                # small oscillating local search around last seen side
                # pattern: try a short pulse toward last_err_sign, check; if still none, flip side, slightly expand
                dir_sign = last_err_sign if (rec_pulses % 2 == 0) else -last_err_sign
                spin_pulse(arlo, REC_PWR, dir_sign, REC_PULSE_MS)

                # check camera right after the pulse
                ok2, frame2 = read_fn()
                if ok2:
                    det2 = detect_marker(frame2, restrict_id=TARGET_ID)
                else:
                    det2 = None

                if det2 is not None:
                    # reacquired -> ALIGN
                    x_px, cx, w = det2["x_px"], det2["cx"], det2["w"]
                    err = cx - (w/2.0)
                    last_err_sign = +1 if err>0 else -1
                    stop(arlo)
                    state = "ALIGN"
                    lost = 0
                    continue

                rec_pulses += 1
                if rec_pulses >= REC_MAX_PULSES:
                    # give up and go back to SEARCH (still no 360)
                    stop(arlo)
                    state = "SEARCH"
                    continue

                # otherwise keep oscillating locally
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
