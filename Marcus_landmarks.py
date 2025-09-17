# find_landmark_search_then_drive_simple.py
import time, numpy as np
from CameraDetection_util import CameraUtils, ArucoUtils
import robot

# ==== Config ====
IMG_W, IMG_H, FPS = 960, 720, 30
F_PX, MARKER_MM = 1275.0, 140.0
TARGET_ID = None

STOP_AT_MM, CENTER_DEADBAND_PX = 420.0, 28
REQUIRED_HITS, LOST_TO_SEARCH = 2, 25
TURN_PWR, BASE_PWR = 50, 60
MAX_PWR, MIN_PWR = 100, 40
Kp_far, Kp_near = 0.10, 0.06
MAX_STEER, EMA_ALPHA = 24, 0.35

# ==== Helpers ====
def clamp_power(p): return max(0, min(MAX_PWR, max(MIN_PWR, int(round(p)))))

_spin = {"on": False, "t0": 0.0}
def spin_pwm_step(arlo, power, period_ms=350, duty=0.22):
    now = time.time()
    if _spin["t0"] == 0.0: _spin.update(t0=now, on=False)
    elapsed = (now - _spin["t0"])*1000.0
    if _spin["on"]:
        if elapsed >= duty*period_ms: arlo.stop(); _spin.update(on=False, t0=now)
    else:
        if elapsed >= (1-duty)*period_ms:
            arlo.go_diff(clamp_power(power), clamp_power(power), 0, 1)  # spin left
            _spin.update(on=True, t0=now)

def estimate_Z_mm(x_px, f_px=F_PX, X_mm=MARKER_MM):
    return (f_px * X_mm) / max(x_px, 1e-6)

# ==== Init ====
arlo = robot.Robot()
cam = CameraUtils(width=IMG_W, height=IMG_H, fx=F_PX, fy=F_PX)
cam.start_camera(width=IMG_W, height=IMG_H, fps=FPS)
aruco = ArucoUtils(marker_length=MARKER_MM/1000.0)

state, hits, lost_frames, err_filt = 0, 0, 0, 0.0
SEARCH, DRIVE = 0, 1
arlo.stop()

try:
    while True:
        # ----- SEARCH rotation -----
        if state == SEARCH:
            spin_pwm_step(arlo, TURN_PWR)
            if not _spin["on"]: time.sleep(0.05)

        ret, frame = cam.get_frame()
        if not ret: continue

        # ----- Detect markers -----
        corners, ids = aruco.detect_markers(frame)
        if corners is not None:
            # pick largest marker (by perimeter)
            markers = [
                {"id": int(mid), "cx": float(c[:,0].mean()), 
                 "x_px": 0.5*(np.linalg.norm(c[0]-c[3]) + np.linalg.norm(c[1]-c[2])),
                 "w": frame.shape[1], "perimeter": sum(np.linalg.norm(c[i]-c[(i+1)%4]) for i in range(4))}
                for c, mid in zip(corners, ids.flatten()) 
                if TARGET_ID is None or int(mid)==TARGET_ID
            ]
            det = max(markers, key=lambda m: m["perimeter"], default=None)
        else:
            det = None

        # ----- SEARCH state -----
        if state == SEARCH:
            if det is None: hits=0; continue
            hits += 1
            if hits < REQUIRED_HITS: continue
            arlo.stop(); state=DRIVE; lost_frames=0; err_filt=0.0
            arlo.go_diff(BASE_PWR, BASE_PWR, 1, 1)
            continue

        # ----- DRIVE state -----
        if det is None:
            lost_frames += 1
            if lost_frames > LOST_TO_SEARCH: arlo.stop(); state=SEARCH; hits=0
            continue
        lost_frames=0

        err = det["cx"] - det["w"]*0.5
        err_filt = (1-EMA_ALPHA)*err_filt + EMA_ALPHA*err
        Z_mm = estimate_Z_mm(det["x_px"])
        Kp = Kp_near if Z_mm<800 else Kp_far
        steer = 0.0 if abs(err_filt)<CENTER_DEADBAND_PX else float(np.clip(Kp*err_filt, -MAX_STEER, MAX_STEER))

        arlo.go_diff(clamp_power(BASE_PWR+steer), clamp_power(BASE_PWR-steer), 1, 1)

        if Z_mm <= STOP_AT_MM:
            arlo.stop()
            print(f"Done: centered and ~{Z_mm:.0f} mm away (ID={det['id']})")
            break

finally:
    cam.stop_camera()
    arlo.stop()
