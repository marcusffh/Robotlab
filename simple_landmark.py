# find_landmark_smooth.py
import time, cv2, numpy as np
import robot

# ==== Camera / marker (same as your current working version) ====
F_PX      = 1275.0
MARKER_MM = 140.0
TARGET_ID = None

IMG_W, IMG_H, FPS  = 960, 720, 30

# ==== Stop and align params ====
STOP_AT_MM     = 420.0         # stop ~0.42 m from marker
CENTER_DEADBAND_PX = 28        # ignore small center errors
LOST_GRACE_FRAMES = 10         # keep coasting this many frames when lost

# ==== Steering control ====
BASE_PWR   = 60                # forward cruise power (>=40 per robot.py)
MAX_PWR    = 100               # cap (<=127 per robot.py)
MIN_PWR    = 40                # never go below 40 (or 0)
Kp_far     = 0.10              # deg/pixel-ish scaling into power delta (far)
Kp_near    = 0.06              # gentler steering when close
MAX_STEER  = 24                # max |delta power| applied to one side
EMA_ALPHA  = 0.35              # smoothing for center error

# ==== Robot + camera ====
arlo = robot.Robot()

def clamp_power(p):
    if p <= 0: return 0
    return max(MIN_PWR, min(MAX_PWR, int(round(p))))

# --- your camera + detection helpers (unchanged) ---
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
        if not cap.isOpened():
            raise RuntimeError(f"Camera init failed: {e}")
        def read_fn(): return cap.read()
        def release_fn():
            try: cap.release()
            except: pass
        return read_fn, release_fn

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
    x_px = 0.5*(v1+v2)
    cx   = float((TL[0]+TR[0]+BR[0]+BL[0]) / 4.0)
    h, w = frame_bgr.shape[:2]
    return {"id": mid, "x_px": float(x_px), "cx": cx, "w": w}

def estimate_Z_mm(x_px, f_px=F_PX, X_mm=MARKER_MM):
    return (f_px * X_mm) / max(x_px, 1e-6)

# ==== Main loop with smooth steering ====
read, release = make_camera()
err_filt = 0.0
lost_frames = 0

try:
    # start rolling slowly (helps reduce startup twitch)
    arlo.go_diff(BASE_PWR, BASE_PWR, 1, 1)

    while True:
        ok, frame = read()
        if not ok:
            # keep previous command; try next frame
            continue

        det = detect_marker(frame, restrict_id=TARGET_ID)

        if det is None:
            # Grace period: keep last steering and count lost frames
            lost_frames += 1
            if lost_frames > LOST_GRACE_FRAMES:
                # gentle search while moving: slight left bias
                L = clamp_power(BASE_PWR - 8)
                R = clamp_power(BASE_PWR + 8)
                arlo.go_diff(L, R, 1, 1)
            continue

        # reset lost counter on sight
        lost_frames = 0

        # center error (pixels): negative = marker left of image center
        err = det["cx"] - (det["w"] * 0.5)
        # low-pass filter the error
        err_filt = (1.0 - EMA_ALPHA) * err_filt + EMA_ALPHA * err

        # distance estimate for gain scheduling
        Z_mm = estimate_Z_mm(det["x_px"])
        Kp = Kp_near if Z_mm < 800.0 else Kp_far  # gentler when we’re closer

        # apply deadband
        if abs(err_filt) < CENTER_DEADBAND_PX:
            steer = 0.0
        else:
            steer = np.clip(Kp * err_filt, -MAX_STEER, MAX_STEER)

        # convert steer to differential power (note sign: err>0 => marker right => steer right)
        L = clamp_power(BASE_PWR + steer)
        R = clamp_power(BASE_PWR - steer)

        # drive forward with steering; no stop/turn twitching
        arlo.go_diff(L, R, 1, 1)  # forward both, different speeds

        # stop condition
        if Z_mm <= STOP_AT_MM:
            arlo.stop()
            print(f"Done: centered and ~{Z_mm:.0f} mm away (ID={det['id']}).")
            break

        # small sleep to keep loop tame (optional; 30 FPS is fine without)
        # time.sleep(0.005)

finally:
    try: release()
    except: pass
    arlo.stop()
