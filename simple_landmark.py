# find_landmark_min2.py
import time, math, cv2, numpy as np
import robot

# ========= Camera / marker =========
F_PX      = 1275.0      # focal length (px) from your Ex.1
MARKER_MM = 140.0       # marker side length in mm
TARGET_ID = None        # set an int to lock a specific marker, else None

# ========= Behavior tuning =========
IMG_W, IMG_H, FPS  = 960, 720, 30
CENTER_TOL_PX      = 36               # how close to center before we drive
STOP_AT_MM         = 420.0            # stop when est. Z <= this (≈0.42 m)
TURN_PWR, FWD_PWR  = 40, 64           # per robot.py: >=40 (or 0)
TURN_DT            = 0.10             # short rotate burst
FWD_DT_FAR         = 0.22             # forward burst when far
FWD_DT_NEAR        = 0.12             # forward burst when near

# ========= Camera (Picamera2 first, GStreamer fallback) =========
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
        cam.configure(cfg); cam.start(); time.sleep(0.9)  # warmup
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
        if not cap.isOpened():
            raise RuntimeError(f"Camera init failed: {e}")
        def read_fn():
            return cap.read()
        def release_fn():
            try: cap.release()
            except: pass
        return read_fn, release_fn

# ========= ArUco detection (robust) =========
def detect_marker(frame_bgr, restrict_id=None):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    params     = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(frame_bgr, dictionary, parameters=params)
    if ids is None or len(corners) == 0:
        return None
    best = None
    for c, mid in zip(corners, ids.flatten()):
        if restrict_id is not None and int(mid) != restrict_id:
            continue
        pts = c.reshape(-1,2)
        per = (np.linalg.norm(pts[0]-pts[1]) + np.linalg.norm(pts[1]-pts[2]) +
               np.linalg.norm(pts[2]-pts[3]) + np.linalg.norm(pts[3]-pts[0]))
        if best is None or per > best[0]:
            best = (per, int(mid), pts)
    if best is None: return None
    _, mid, pts = best
    TL, TR, BR, BL = pts
    v1 = np.linalg.norm(TL - BL)
    v2 = np.linalg.norm(TR - BR)
    x_px = 0.5*(v1+v2)                               # vertical-edge avg (stable under tilt)
    cx   = float((TL[0]+TR[0]+BR[0]+BL[0]) / 4.0)    # marker center x
    h, w = frame_bgr.shape[:2]
    return {"id": mid, "x_px": float(x_px), "cx": cx, "w": w}

def estimate_Z_mm(x_px, f_px=F_PX, X_mm=MARKER_MM):
    return (f_px * X_mm) / max(x_px, 1e-6)

# ========= Motion helpers using robot.Robot =========
arlo = robot.Robot()

def step_turn(left=True, dt=TURN_DT):
    if left: arlo.go_diff(TURN_PWR, TURN_PWR, 0, 1)
    else:    arlo.go_diff(TURN_PWR, TURN_PWR, 1, 0)
    time.sleep(dt); arlo.stop()

def step_forward(dt):
    arlo.go_diff(FWD_PWR, FWD_PWR, 1, 1); time.sleep(dt); arlo.stop()

# ========= Main loop =========
read, release = make_camera()
try:
    while True:
        ok, frame = read()
        if not ok:
            step_turn(left=True);  # keep spinning slowly if a frame hiccups
            continue

        det = detect_marker(frame, restrict_id=TARGET_ID)
        if det is None:
            step_turn(left=True)   # search spin
            continue

        err_px = det["cx"] - (det["w"] * 0.5)
        if abs(err_px) > CENTER_TOL_PX:
            step_turn(left = (err_px < 0))  # if marker left of center -> turn left
            continue

        Z_mm = estimate_Z_mm(det["x_px"], F_PX, MARKER_MM)
        if Z_mm > STOP_AT_MM:
            step_forward(FWD_DT_FAR if Z_mm > 1.7*STOP_AT_MM else FWD_DT_NEAR)
        else:
            print(f"Done: centered and ~{Z_mm:.0f} mm away (ID={det['id']}).")
            break
finally:
    try: release()
    except: pass
    arlo.stop()
