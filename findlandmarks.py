#!/usr/bin/env python3
"""
find_landmark_steps.py
Stepwise approach using CalibratedRobot only:
SEARCH → ALIGN (micro turns) → APPROACH (alternate tiny correction & straight step) → DONE.

Vision: our existing ArUco detection + Z-from-pixels (no ArucoUtils).
Motion: ONLY CalibratedRobot.turn_angle / drive_distance / stop (no raw wheel power).
"""

import time, sys, os, select, signal
import numpy as np

# ========= Camera / marker =========
F_PX        = 1275.0     # focal length in px (your calibration)
MARKER_MM   = 140.0      # 14 cm tag
TARGET_ID   = None       # lock to a specific id or leave None

# ========= Search / align tuning =========
SEARCH_STEP_DEG        = 8.0      # small rotate pulse per search step
SEARCH_SLEEP_S         = 0.06
PX_TOL                 = 30       # center deadband in pixels
PX_KP_DEG_PER_PX       = 0.05     # +err (marker right) -> NEG turn (right)
MAX_ALIGN_STEP_DEG     = 10.0

# ========= Forward step policy =========
STOP_AT_MM             = 50.0     # 5 cm stand-off
STEP_MIN_M             = 0.08
STEP_MAX_M             = 0.35
STEP_SCALE             = 0.60     # step ≈ remaining(m) * SCALE, clamped to [MIN, MAX]

# ========= Robustness =========
LOST_LIMIT             = 12
LOOP_SLEEP_S           = 0.04

# ========= Safe abort =========
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

# ========= Bring in CalibratedRobot =========
try:
    from Exercise1.CalibratedRobot import CalibratedRobot
except ModuleNotFoundError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "exercise1"))
    from Exercise1.CalibratedRobot import CalibratedRobot

# ========= Camera (Picamera2 first, GStreamer fallback) =========
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

# ========= Vision (our detector) =========
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
    # pinhole: Z = f*X/x
    return (f_px * X_mm) / max(x_px, 1e-6)

# ========= Motion helpers (CalibratedRobot) =========
def rotate_step(bot: CalibratedRobot, deg: float, speed=None):
    """Turn a small angle. Positive deg = left, negative = right (CalibratedRobot convention)."""
    if abs(deg) < 0.1:
        return
    bot.turn_angle(deg, speed=speed)

def forward_step(bot: CalibratedRobot, meters: float, speed=None):
    """Drive straight forward for 'meters'."""
    if meters <= 0.0:
        return
    bot.drive_distance(meters, direction=bot.FORWARD, speed=speed)

# ========= Control logic =========
def choose_turn_deg(err_px: float) -> float:
    """
    err_px = cx - w/2 ; positive => marker appears to the RIGHT.
    To center, we must turn RIGHT ⇒ NEGATIVE degrees for CalibratedRobot.turn_angle.
    """
    step = -(err_px * PX_KP_DEG_PER_PX)
    return float(np.clip(step, -MAX_ALIGN_STEP_DEG, MAX_ALIGN_STEP_DEG))

def choose_step_m(Z_mm: float) -> float:
    remaining_m = max(0.0, Z_mm - STOP_AT_MM) / 1000.0
    step = remaining_m * STEP_SCALE
    return float(np.clip(step, STEP_MIN_M, STEP_MAX_M))

def main():
    # Robot + Camera
    bot = CalibratedRobot()
    read_fn, release_fn = make_camera()

    state     = "SEARCH"
    lost      = 0
    need_turn = True  # APPROACH alternates: small correction -> straight forward step

    print("find_landmark_steps: SEARCH → ALIGN → APPROACH → DONE (CalibratedRobot only)")
    try:
        while True:
            if ABORT or user_requested_quit():
                print("[ABORT] user stop"); break

            ok, frame = read_fn()
            if not ok:
                bot.stop()
                time.sleep(0.05)
                continue

            det = detect_marker(frame, restrict_id=TARGET_ID)

            # ===== SEARCH =====
            if state == "SEARCH":
                if det is None:
                    print(f"[SEARCH] rotate {SEARCH_STEP_DEG:.1f}° then check…")
                    rotate_step(bot, +SEARCH_STEP_DEG)  # small left pulse
                    time.sleep(SEARCH_SLEEP_S)
                    continue

                # Found
                x_px, cx, w = det["x_px"], det["cx"], det["w"]
                err_px = cx - (w * 0.5)
                Z_mm   = estimate_Z_mm(x_px)
                print(f"[FOUND] id={det['id']}  Z≈{Z_mm:.0f} mm  size={x_px:.1f}px  err={err_px:.1f}px")
                state = "ALIGN"; lost = 0
                continue

            # ===== LOSS HANDLING (other states) =====
            if det is None:
                lost += 1
                print(f"[LOST] no detection ({lost}/{LOST_LIMIT})")
                if lost >= LOST_LIMIT:
                    print("[LOST] returning to SEARCH")
                    state = "SEARCH"; lost = 0
                time.sleep(LOOP_SLEEP_S)
                continue
            else:
                lost = 0

            # Parse detection
            cx, w = det["cx"], det["w"]
            err_px = cx - (w * 0.5)  # +: marker right of center
            x_px   = det["x_px"]
            Z_mm   = estimate_Z_mm(x_px)

            # ===== ALIGN =====
            if state == "ALIGN":
                if abs(err_px) <= PX_TOL:
                    print(f"[ALIGN→APPROACH] centered; Z≈{Z_mm:.0f} mm")
                    state = "APPROACH"; need_turn = True
                else:
                    turn_deg = choose_turn_deg(err_px)  # NEG when marker is right -> turn right
                    print(f"[ALIGN] err={err_px:.1f}px → turn {turn_deg:.1f}°")
                    rotate_step(bot, turn_deg)
                    time.sleep(SEARCH_SLEEP_S)
                continue

            # ===== APPROACH (alternate tiny correction & straight step) =====
            if state == "APPROACH":
                if Z_mm <= STOP_AT_MM:
                    bot.stop()
                    print(f"[DONE] close enough: Z≈{Z_mm:.0f} mm (≤ {STOP_AT_MM:.0f} mm)")
                    break

                if need_turn and abs(err_px) > PX_TOL:
                    turn_deg = choose_turn_deg(err_px)
                    print(f"[MOVE] correcting: turn {turn_deg:.1f}°  (err={err_px:.1f}px, Z≈{Z_mm:.0f} mm)")
                    rotate_step(bot, turn_deg)
                    need_turn = False
                    time.sleep(SEARCH_SLEEP_S)
                    continue

                step_m = choose_step_m(Z_mm)
                print(f"[MOVE] forward {step_m:.2f} m  (Z≈{Z_mm:.0f} mm)")
                forward_step(bot, step_m)
                need_turn = True
                time.sleep(SEARCH_SLEEP_S)
                continue

            time.sleep(LOOP_SLEEP_S)

    except KeyboardInterrupt:
        print("\n[ABORT] Ctrl-C")
    finally:
        try: bot.stop()
        except: pass
        release_fn()
        print("Exiting.")

if __name__ == "__main__":
    main()
