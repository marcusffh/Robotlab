#!/usr/bin/env python3
"""
find_landmark_arlo.py — smooth visual-servo approach (no 360, no jerks)
"""

import time, sys, signal, select
import numpy as np

# ===== Camera / marker =====
F_PX        = 1275.0
MARKER_MM   = 140.0
TARGET_ID   = None

# ===== Drive / calibration =====
MIN_PWR     = 40
MAX_PWR     = 127
CAL_KL      = 0.98   # match your straight-driver
CAL_KR      = 1.00

# ===== Behavior tuning =====
# SEARCH
SEARCH_PWR      = 44
SEARCH_DIR      = +1
SPIN_PERIOD_MS  = 350
SPIN_DUTY       = 0.22

# ALIGN
TURN_PWR        = 42
ALIGN_PULSE_MS  = 90
PX_TOL          = 30

# TRACK_DRIVE
BASE_BIAS       = 0.0      # <-- was -0.06 (caused right snap)
EMA_ALPHA       = 0.30
Kp              = 0.0008
Ki              = 0.00003  # tiny bump so it trims residual drift
Kd              = 0.0018
BIAS_MAX        = 0.40
STEER_DEADBAND  = 0.02     # new: keep tiny steer = 0
DRIVE_PWR_MIN   = 46
DRIVE_PWR_MAX   = 68
NEAR_MM_SLOW    = 900.0
STEP_DT         = 0.06
SLEW_PER_STEP   = 10

STOP_AT_MM      = 450.0
DONE_MM_HYST    = 30.0

# LOST handling
LOST_LIMIT      = 10
REC_PWR         = 44
REC_PULSE_MS    = 110
REC_MAX_PULSES  = 24

# ===== Safe abort =====
ABORT = False
def _sig(*_):
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

def drive_set(arlo, left_power, right_power, fwd=True):
    L, R = scaled_lr(left_power, right_power)
    if fwd: arlo.go_diff(L, R, 1, 1)
    else:   arlo.go_diff(L, R, 0, 0)

def spin_continuous(arlo, power, dir_sign):
    power = max(MIN_PWR, min(MAX_PWR, int(power)))
    L,R = scaled_lr(power, power)
    if dir_sign >= 0: arlo.go_diff(L, R, 1, 0)  # right spin
    else:             arlo.go_diff(L, R, 0, 1)  # left spin

def stop(arlo): arlo.stop()

# ===== Duty-cycled slow spin for SEARCH =====
_spin_state = {"on": False, "t0": 0.0}
def spin_pwm_step(arlo, power, dir_sign, period_ms=SPIN_PERIOD_MS, duty=SPIN_DUTY):
    now = time.time()
    if _spin_state["t0"] == 0.0:
        _spin_state["t0"] = now
        _spin_state["on"] = False
    elapsed = (now - _spin_state["t0"]) * 1000.0
    on_ms   = duty * period_ms
    off_ms  = (1.0 - duty) * period_ms
    if _spin_state["on"]:
        if elapsed >= on_ms:
            stop(arlo); _spin_state["on"] = False; _spin_state["t0"] = now
    else:
        if elapsed >= off_ms:
            spin_continuous(arlo, power, dir_sign); _spin_state["on"] = True; _spin_state["t0"] = now

# Smoothing helper
def slew_toward(prev, target, max_step):
    if target > prev + max_step: return prev + max_step
    if target < prev - max_step: return prev - max_step
    return target

# ===== Main =====
def main():
    import robot as rb
    arlo = rb.Robot()
    read_fn, release_fn = make_camera()

    state = "SEARCH"
    lost = 0
    err_filt = 0.0
    err_prev = 0.0
    err_int  = 0.0
    last_err_sign = +1
    rec_pulses = 0

    locked_travel_mm = None
    remaining_mm     = None

    last_L = 0
    last_R = 0

    print("SM: SEARCH(duty) -> ALIGN -> LOCK -> TRACK_DRIVE -> RECOVER(no 360) -> DONE")
    try:
        loop_t_prev = time.time()
        while True:
            if ABORT or user_requested_quit():
                print("[ABORT] user stop"); break

            # pace loop
            now = time.time()
            dt = now - loop_t_prev
            if dt < STEP_DT:
                time.sleep(STEP_DT - dt)
                now = time.time()
                dt = now - loop_t_prev
            loop_t_prev = now

            ok, frame = read_fn()
            if not ok:
                stop(arlo)
                continue

            # ===== SEARCH =====
            if state == "SEARCH":
                spin_pwm_step(arlo, SEARCH_PWR, SEARCH_DIR)
                det = detect_marker(frame, restrict_id=TARGET_ID)
                if det is None:
                    continue
                stop(arlo)
                state = "ALIGN"
                lost = 0
                locked_travel_mm = None
                remaining_mm = None
            else:
                det = detect_marker(frame, restrict_id=TARGET_ID)

            # ===== detection / loss =====
            if det is None:
                lost += 1
                if state in ("ALIGN", "LOCK", "TRACK_DRIVE"):
                    if lost >= LOST_LIMIT:
                        state = "RECOVER"
                        rec_pulses = 0
                        stop(arlo)
                        continue
                    if state == "TRACK_DRIVE" and last_L>0 and last_R>0:
                        L_t = max(DRIVE_PWR_MIN, int(0.7*last_L))
                        R_t = max(DRIVE_PWR_MIN, int(0.7*last_R))
                        L = slew_toward(last_L, L_t, SLEW_PER_STEP)
                        R = slew_toward(last_R, R_t, SLEW_PER_STEP)
                        drive_set(arlo, L, R, True)
                        last_L, last_R = L, R
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
                    state = "LOCK"
                    continue
                spin_continuous(arlo, TURN_PWR, +1 if err>0 else -1)
                time.sleep(ALIGN_PULSE_MS/1000.0)
                stop(arlo)
                continue

            if state == "LOCK":
                D = max(0.0, Z - STOP_AT_MM)
                locked_travel_mm = D
                remaining_mm = D
                err_filt = err
                err_prev = err
                err_int  = 0.0
                state = "TRACK_DRIVE"

            if state == "TRACK_DRIVE":
                # Filtered error
                err_filt = (1.0 - EMA_ALPHA) * err_filt + EMA_ALPHA * err
                derr = (err_filt - err_prev) / max(dt, 1e-6)
                err_prev = err_filt

                # Integral (bounded)
                if abs(err_filt) > PX_TOL:
                    err_int += err_filt * dt
                    err_int = max(min(err_int, 2000.0), -2000.0)
                else:
                    err_int *= 0.9

                # --- Fix 1: kill derivative when centered to avoid lock snap
                if abs(err_filt) <= PX_TOL:
                    derr = 0.0

                steer = BASE_BIAS + Kp*err_filt + Ki*err_int + Kd*derr

                # --- Fix 2: steer deadband
                if -STEER_DEADBAND < steer < STEER_DEADBAND:
                    steer = 0.0

                steer = max(min(steer, BIAS_MAX), -BIAS_MAX)

                # Forward speed schedule
                dist_scale = 1.0
                if remaining_mm is not None:
                    if remaining_mm < NEAR_MM_SLOW:
                        dist_scale = max(0.35, remaining_mm / NEAR_MM_SLOW)
                else:
                    if Z < NEAR_MM_SLOW:
                        dist_scale = max(0.35, Z / NEAR_MM_SLOW)

                base_power_target = DRIVE_PWR_MIN + (DRIVE_PWR_MAX - DRIVE_PWR_MIN) * dist_scale

                L_target = base_power_target * (1.0 - steer)
                R_target = base_power_target * (1.0 + steer)

                L = slew_toward(last_L, int(L_target), SLEW_PER_STEP)
                R = slew_toward(last_R, int(R_target), SLEW_PER_STEP)

                drive_set(arlo, L, R, True)
                last_L, last_R = L, R

                # Telemetry (helps verify no hidden bias)
                print(f"err={err_filt:+6.1f}px steer={steer:+.3f} L={L:3d} R={R:3d} Z={Z:5.0f}mm rem={remaining_mm if remaining_mm is not None else -1:.0f}")

                # Remaining travel update
                if remaining_mm is not None:
                    if not hasattr(main, "_Z_prev"):
                        main._Z_prev = Z
                    dZ = max(0.0, main._Z_prev - Z)
                    remaining_mm = max(0.0, remaining_mm - dZ)
                    main._Z_prev = Z
                    if remaining_mm <= 10.0 or Z <= (STOP_AT_MM + DONE_MM_HYST):
                        stop(arlo)
                        print(f"[DONE] stop at Z={Z:.0f} mm (planned D={locked_travel_mm:.0f} mm)")
                        break
                else:
                    if Z <= (STOP_AT_MM + DONE_MM_HYST):
                        stop(arlo)
                        print(f"[DONE] stop at Z={Z:.0f} mm")
                        break
                continue

            if state == "RECOVER":
                dir_sign = last_err_sign if (rec_pulses % 2 == 0) else -last_err_sign
                spin_continuous(arlo, REC_PWR, dir_sign)
                time.sleep(REC_PULSE_MS/1000.0)
                stop(arlo)

                ok2, frame2 = read_fn()
                det2 = detect_marker(frame2, restrict_id=TARGET_ID) if ok2 else None
                if det2 is not None:
                    x_px, cx, w = det2["x_px"], det2["cx"], det2["w"]
                    Z = estimate_Z_mm(x_px)
                    err = cx - (w/2.0)
                    last_err_sign = +1 if err>0 else -1
                    state = "ALIGN"
                    lost = 0
                    continue

                rec_pulses += 1
                if rec_pulses >= REC_MAX_PULSES:
                    state = "SEARCH"
                    last_L, last_R = 0, 0
                    stop(arlo)
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
