#!/usr/bin/env python3
"""
find_landmark_arlo.py — smooth visual-servo approach (no 360, no jerks)

Workflow:
  1) SEARCH: very slow duty-cycled rotation while reading camera every frame.
  2) ALIGN : gentle micro-rotations to center marker (deadband).
  3) LOCK  : once centered, estimate Z and "lock" target travel distance D = Z - STOP_AT_MM.
  4) TRACK_DRIVE: continuous forward motion with smooth steering (PID-like), no jerky stop/go.
                  We integrate the distance we’ve traveled (from Z updates) until D is used up.
                  If the marker is momentarily lost, we keep gentle motion; if lost for long -> RECOVER.
  5) RECOVER: small oscillating nudges toward last-seen side until reacquired (no 360).
  6) DONE : stop at stand-off distance.

Controls:
  - Ctrl-C            -> safe stop
  - Type 'q' + Enter  -> safe stop (SSH-friendly)

Requirements:
  - robot.py (Robot.go_diff(...), Robot.stop())
  - OpenCV (aruco)
  - picamera2 preferred, GStreamer fallback (headless)
"""

import time, sys, signal, select
import numpy as np

# ===== Camera / marker =====
F_PX        = 1275.0      # your calibrated focal length (px)
MARKER_MM   = 140.0       # marker height (mm)
TARGET_ID   = None        # lock to specific ArUco id (e.g., 6) or None

# ===== Drive / calibration =====
MIN_PWR     = 40
MAX_PWR     = 127
CAL_KL      = 0.98        # tuned scales (left)
CAL_KR      = 1.00        # tuned scales (right)

# ===== Behavior tuning =====
# SEARCH: duty-cycled spin for truly slow rotation
SEARCH_PWR      = 44
SEARCH_DIR      = +1       # +1 spin right, -1 spin left
SPIN_PERIOD_MS  = 350
SPIN_DUTY       = 0.22

# ALIGN
TURN_PWR        = 42
ALIGN_PULSE_MS  = 90
PX_TOL          = 30       # center deadband (px)

# TRACK_DRIVE (smooth, no jerks)
BASE_BIAS       = -0.06    # cancels systematic left drift
EMA_ALPHA       = 0.30     # low-pass for pixel error
Kp              = 0.0008   # proportional steering gain (px -> bias)
Ki              = 0.00002  # tiny integral term for slow residual bias
Kd              = 0.0018   # small differential term (on filtered error)
BIAS_MAX        = 0.40     # clamp steering bias
DRIVE_PWR_MIN   = 46       # min forward power while tracking
DRIVE_PWR_MAX   = 68       # max forward power while tracking
NEAR_MM_SLOW    = 900.0    # within this distance, ramp down speed
STEP_DT         = 0.06     # control loop period target (s)
SLEW_PER_STEP   = 10       # max change in power per loop (slew-rate limit)

STOP_AT_MM      = 450.0    # stand-off
DONE_MM_HYST    = 30.0     # small hysteresis to avoid stop/restart

# LOST handling
LOST_LIMIT      = 10       # consecutive frames without detection -> RECOVER
REC_PWR         = 44
REC_PULSE_MS    = 110
REC_MAX_PULSES  = 24

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

def drive_set(arlo, left_power, right_power, fwd=True):
    """Set continuous wheel powers with calibration."""
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

# ===== Slew limiting (avoid jerky power jumps) =====
def slew toward(prev, target, max_step):
    if target > prev + max_step: return prev + max_step
    if target < prev - max_step: return prev - max_step
    return target

# (Python identifiers can’t have spaces—rename properly.)
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

    # distance locking
    locked_travel_mm = None   # target travel distance once locked (Z0 - STOP_AT_MM)
    remaining_mm     = None

    # power smoothing
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

            # ===== SEARCH: slow rotate and read each frame =====
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
                # fall through with det
            else:
                det = detect_marker(frame, restrict_id=TARGET_ID)

            # ===== detection / loss =====
            if det is None:
                lost += 1
                if state in ("ALIGN", "LOCK", "TRACK_DRIVE"):
                    if lost >= LOST_LIMIT:
                        # enter local RECOVER using last seen side
                        state = "RECOVER"
                        rec_pulses = 0
                        stop(arlo)
                        continue
                    # short grace: keep current drive (if any), but reduce power
                    if state == "TRACK_DRIVE" and last_L>0 and last_R>0:
                        # gently coast with reduced power while searching in place
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
                # gentle micro-pulse toward error sign
                spin_continuous(arlo, TURN_PWR, +1 if err>0 else -1)
                time.sleep(ALIGN_PULSE_MS/1000.0)
                stop(arlo)
                continue

            if state == "LOCK":
                # Estimate initial distance and lock a travel budget
                # D = max(0, Z - STOP_AT_MM). We'll integrate remaining_mm downward using Z updates.
                D = max(0.0, Z - STOP_AT_MM)
                locked_travel_mm = D
                remaining_mm = D
                # reset controller terms
                err_filt = err
                err_prev = err
                err_int  = 0.0
                state = "TRACK_DRIVE"
                # fall through to drive this loop

            if state == "TRACK_DRIVE":
                # Update filtered error
                err_filt = (1.0 - EMA_ALPHA) * err_filt + EMA_ALPHA * err
                derr = (err_filt - err_prev) / max(dt, 1e-6)
                err_prev = err_filt
                # bounded integral (anti-windup)
                if abs(err_filt) > PX_TOL:
                    err_int += err_filt * dt
                    err_int = max(min(err_int, 2000.0), -2000.0)
                else:
                    err_int *= 0.9  # decay in deadband

                # steering bias: base + PID on (filtered) pixel error
                steer = BASE_BIAS + Kp*err_filt + Ki*err_int + Kd*derr
                steer = max(min(steer, BIAS_MAX), -BIAS_MAX)

                # forward speed schedule (softer near goal)
                # map remaining_mm (or Z) to power
                dist_scale = 1.0
                if remaining_mm is not None:
                    if remaining_mm < NEAR_MM_SLOW:
                        dist_scale = max(0.35, remaining_mm / NEAR_MM_SLOW)  # 0.35..1
                else:
                    if Z < NEAR_MM_SLOW:
                        dist_scale = max(0.35, Z / NEAR_MM_SLOW)

                base_power_target = DRIVE_PWR_MIN + (DRIVE_PWR_MAX - DRIVE_PWR_MIN) * dist_scale

                # compute target L/R from bias
                L_target = base_power_target * (1.0 - steer)
                R_target = base_power_target * (1.0 + steer)

                # slew limit to avoid jerks
                L = slew_toward(last_L, int(L_target), SLEW_PER_STEP)
                R = slew_toward(last_R, int(R_target), SLEW_PER_STEP)

                drive_set(arlo, L, R, True)
                last_L, last_R = L, R

                # Update remaining travel using Z (distance-to-go reduction)
                if remaining_mm is not None:
                    # How much closer did we get this loop? Use previous Z minus new Z (monotonic, clamp)
                    # We need Z_prev; keep it in closure:
                    if not hasattr(main, "_Z_prev"):
                        main._Z_prev = Z
                    dZ = max(0.0, main._Z_prev - Z)  # mm closed this cycle
                    remaining_mm = max(0.0, remaining_mm - dZ)
                    main._Z_prev = Z

                    # finish conditions (either remaining_mm near 0, or Z <= STOP_AT_MM+HYST)
                    if remaining_mm <= 10.0 or Z <= (STOP_AT_MM + DONE_MM_HYST):
                        stop(arlo)
                        print(f"[DONE] stop at Z={Z:.0f} mm (planned D={locked_travel_mm:.0f} mm)")
                        break
                else:
                    # fallback: use absolute Z threshold
                    if Z <= (STOP_AT_MM + DONE_MM_HYST):
                        stop(arlo)
                        print(f"[DONE] stop at Z={Z:.0f} mm")
                        break

                continue

            if state == "RECOVER":
                # Oscillating nudges around last-seen side (no 360)
                dir_sign = last_err_sign if (rec_pulses % 2 == 0) else -last_err_sign
                # small pulse, then check
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
                    # re-enter TRACK path via ALIGN to re-center quickly
                    state = "ALIGN"
                    lost = 0
                    continue

                rec_pulses += 1
                if rec_pulses >= REC_MAX_PULSES:
                    # fall back to SEARCH if we can’t reacquire locally
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
