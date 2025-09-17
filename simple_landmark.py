# find_landmark_boot_pulse_search.py
import time, cv2, numpy as np
import robot

# ==== Camera / marker ====
F_PX      = 1275.0
MARKER_MM = 140.0
TARGET_ID = None

IMG_W, IMG_H, FPS  = 960, 720, 30

# ==== Behavior ====
STOP_AT_MM           = 420.0
CENTER_DEADBAND_PX   = 28
REQUIRED_HITS        = 2       # consecutive detections before leaving SEARCH
LOST_GRACE_FRAMES    = 10
LOST_TO_SEARCH       = 25

# ==== Motion / control (DRIVE stays as you liked it) ====
TURN_PWR   = 50                # power for in-place spin
BASE_PWR   = 60                # forward cruise power
MAX_PWR    = 100; MIN_PWR = 40
Kp_far     = 0.10; Kp_near = 0.06
MAX_STEER  = 24
EMA_ALPHA  = 0.35

# ==== SEARCH pulse-timing (new) ====
TURN_PULSE_DT = 0.10           # rotate for 100 ms
SNAP_DT       = 0.06           # stop for 60 ms to capture a sharp frame

def clamp_power(p):
    if p <= 0: return 0
    return max(MIN_PWR, min(MAX_PWR, int(round(p))))

# ==== Camera ====
def make_camera(width=IMG_W, height=IMG_H, fps=FPS):
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        frame_us = int(1.0/fps * 1_000_000)
        cfg = cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameDurationLimits": (frame_us, frame_us)},
            queue=False
        )
        cam.configure(cfg); cam.start(); time.sleep(0.9)
        def read_fn():
            rgb = cam.capture_array("main")
            return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        def release_fn():
            try: cam.stop()
            except: pass
        return read_fn, release_fn
    except Exception as e:
        def gst(w,h,f):
            return ("libcamerasrc ! videobox autocrop=true ! "
                    f"video/x-raw, width=(int){w}, height=(int){h}, framerate=(fraction){f}/1 ! "
                    "videoconvert ! appsink")
        cap = cv2.VideoCapture(gst(width, height, fps), cv2.CAP_GSTREAMER)
        if not cap.isOpened(): raise RuntimeError(f"Camera init failed: {e}")
        def read_fn(): return cap.read()
        def release_fn():
            try: cap.release()
            except: pass
        return read_fn, release_fn

# ==== Detection ====
def detect_marker(frame_bgr, restrict_id=None):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X_6_250)  # OpenCV constant name
    params     = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(frame_bgr, dictionary, parameters=params)
    if ids is None or len(corners) == 0: return None
    best=None
    for c, mid in zip(corners, ids.flatten()):
        if restrict_id is not None and int(mid) != restrict_id: continue
        pts = c.reshape(-1,2)
        per = (np.linalg.norm(pts[0]-pts[1]) + np.linalg.norm(pts[1]-pts[2]) +
               np.linalg.norm(pts[2]-pts[3]) + np.linalg.norm(pts[3]-pts[0]))
        if best is None or per > best[0]: best = (per, int(mid), pts)
    if best is None: return None
    _, mid, pts = best
    TL, TR, BR, BL = pts
    v1 = np.linalg.norm(TL - BL); v2 = np.linalg.norm(TR - BR)
    x_px = 0.5*(v1+v2)  # size proxy
    cx   = float((TL[0]+TR[0]+BR[0]+BL[0]) / 4.0)
    h, w = frame_bgr.shape[:2]
    return {"id": mid, "x_px": float(x_px), "cx": cx, "w": w}

def estimate_Z_mm(x_px, f_px=F_PX, X_mm=MARKER_MM):
    return (f_px * X_mm) / max(x_px, 1e-6)

# ==== Main ====
arlo = robot.Robot()
read, release = make_camera()

SEARCH, DRIVE = 0, 1
state = SEARCH
hits = 0
lost_frames = 0
err_filt = 0.0

try:
    # Start stationary
    arlo.stop()

    while True:
        # ----- SEARCH: rotate -> stop -> capture -> repeat -----
        if state == SEARCH:
            # rotate in place for a short pulse
            arlo.go_diff(TURN_PWR, TURN_PWR, 0, 1)  # rotate left on center
            time.sleep(TURN_PULSE_DT)
            # stop and give the camera a still moment
            arlo.stop()
            time.sleep(SNAP_DT)

        # Capture a frame (works for both SEARCH and DRIVE)
        ok, frame = read()
        if not ok:
            # if frame hiccup, just loop; SEARCH pulses will continue
            continue

        det = detect_marker(frame, restrict_id=TARGET_ID)

        if state == SEARCH:
            if det is None:
                hits = 0
                continue
            hits += 1
            if hits < REQUIRED_HITS:
                continue
            # lock on: switch to DRIVE (do NOT modify your drive behavior)
            state = DRIVE
            lost_frames = 0
            err_filt = 0.0
            # begin driving forward smoothly
            arlo.go_diff(BASE_PWR, BASE_PWR, 1, 1)
            continue

        # ----- DRIVE (unchanged smooth controller you liked) -----
        if det is None:
            lost_frames += 1
            if lost_frames > LOST_TO_SEARCH:
                arlo.stop()
                state = SEARCH
                hits = 0
            # else: keep last speeds (coast)
            continue

        lost_frames = 0

        # steering around center while moving
        err = det["cx"] - (det["w"] * 0.5)
        err_filt = (1.0 - EMA_ALPHA) * err_filt + EMA_ALPHA * err

        Z_mm = estimate_Z_mm(det["x_px"])
        Kp = Kp_near if Z_mm < 800.0 else Kp_far

        if abs(err_filt) < CENTER_DEADBAND_PX:
            steer = 0.0
        else:
            steer = float(np.clip(Kp * err_filt, -MAX_STEER, MAX_STEER))

        L = clamp_power(BASE_PWR + steer)
        R = clamp_power(BASE_PWR - steer)
        arlo.go_diff(L, R, 1, 1)

        if Z_mm <= STOP_AT_MM:
            arlo.stop()
            print(f"Done: centered and ~{Z_mm:.0f} mm away (ID={det['id']}).")
            break

finally:
    try: release()
    except: pass
    arlo.stop()
