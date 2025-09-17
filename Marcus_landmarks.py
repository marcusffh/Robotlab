# find_landmark_search_then_drive_updated.py
import time
import numpy as np
from CameraDetect_util import CameraUtils, ArucoUtils
import robot
import cv2

# ==== Config ====
F_PX      = 1275.0
MARKER_MM = 140.0
TARGET_ID = None

IMG_W, IMG_H, FPS  = 960, 720, 30

STOP_AT_MM           = 420.0
CENTER_DEADBAND_PX   = 28
REQUIRED_HITS        = 2
LOST_GRACE_FRAMES    = 10
LOST_TO_SEARCH       = 25

TURN_PWR   = 50
BASE_PWR   = 60
MAX_PWR, MIN_PWR = 100, 40
Kp_far, Kp_near = 0.10, 0.06
MAX_STEER  = 24
EMA_ALPHA  = 0.35

# ==== Helper functions ====
def clamp_power(p):
    if p <= 0: return 0
    return max(MIN_PWR, min(MAX_PWR, int(round(p))))

_spin = {"on": False, "t0": 0.0}
def spin_pwm_step(arlo, power, period_ms=350, duty=0.22):
    now = time.time()
    if _spin["t0"] == 0.0:
        _spin["t0"] = now; _spin["on"] = False
    elapsed = (now - _spin["t0"]) * 1000.0
    on_ms   = duty * period_ms
    off_ms  = (1.0 - duty) * period_ms
    if _spin["on"]:
        if elapsed >= on_ms:
            arlo.stop()
            _spin["on"] = False
            _spin["t0"] = now
    else:
        if elapsed >= off_ms:
            L = clamp_power(power); R = clamp_power(power)
            arlo.go_diff(L, R, 0, 1)  # spin left
            _spin["on"] = True
            _spin["t0"] = now

def estimate_Z_mm(x_px, f_px=F_PX, X_mm=MARKER_MM):
    return (f_px * X_mm) / max(x_px, 1e-6)

# ==== Init Robot + Camera + Aruco ====
arlo = robot.Robot()
cam = CameraUtils(width=IMG_W, height=IMG_H, fx=F_PX, fy=F_PX)
cam.start_camera(FPS)
aruco = ArucoUtils(marker_length=MARKER_MM/1000.0)  # convert mm → m

SEARCH, DRIVE = 0, 1
state = SEARCH
hits = 0
lost_frames = 0
err_filt = 0.0

try:
    arlo.stop()  # start stationary

    while True:
        # ----- SEARCH behavior -----
        if state == SEARCH:
            spin_pwm_step(arlo, TURN_PWR)
            if not _spin["on"]:
                time.sleep(0.05)  # SNAP_GUARD_MS

        ret, frame = cam.get_frame()
        if not ret:
            continue

        corners, ids = aruco.detect_markers(frame)
        det = None
        if ids is not None and len(corners) > 0:
            for c, mid in zip(corners, ids.flatten()):
                if TARGET_ID is not None and mid != TARGET_ID:
                    continue
                pts = c.reshape(-1, 2)
                per = (np.linalg.norm(pts[0]-pts[1]) + np.linalg.norm(pts[1]-pts[2]) +
                       np.linalg.norm(pts[2]-pts[3]) + np.linalg.norm(pts[3]-pts[0]))
                # pick largest marker
                if det is None or per > det["perimeter"]:
                    cx = float(np.mean(pts[:,0]))
                    x_px = 0.5 * (np.linalg.norm(pts[0]-pts[3]) + np.linalg.norm(pts[1]-pts[2]))
                    det = {"id": int(mid), "cx": cx, "x_px": x_px, "w": frame.shape[1], "perimeter": per}

        # ----- SEARCH state -----
        if state == SEARCH:
            if det is None:
                hits = 0
                continue
            hits += 1
            if hits < REQUIRED_HITS:
                continue
            arlo.stop()
            state = DRIVE
            lost_frames = 0
            err_filt = 0.0
            arlo.go_diff(BASE_PWR, BASE_PWR, 1, 1)
            continue

        # ----- DRIVE state -----
        if det is None:
            lost_frames += 1
            if lost_frames > LOST_TO_SEARCH:
                arlo.stop()
                state = SEARCH
                hits = 0
            continue

        lost_frames = 0

        # horizontal error
        err = det["cx"] - (det["w"] * 0.5)
        err_filt = (1.0 - EMA_ALPHA) * err_filt + EMA_ALPHA * err

        Z_mm = estimate_Z_mm(det["x_px"])
        Kp = Kp_near if Z_mm < 800.0 else Kp_far

        steer = 0.0 if abs(err_filt) < CENTER_DEADBAND_PX else float(np.clip(Kp * err_filt, -MAX_STEER, MAX_STEER))

        L = clamp_power(BASE_PWR + steer)
        R = clamp_power(BASE_PWR - steer)
        arlo.go_diff(L, R, 1, 1)

        if Z_mm <= STOP_AT_MM:
            arlo.stop()
            print(f"Done: centered and ~{Z_mm:.0f} mm away (ID={det['id']}).")
            break

finally:
    cam.stop_camera()
    arlo.stop()
