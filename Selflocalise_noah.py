#!/usr/bin/env python3
"""
Drive to the midpoint between ArUco markers 6 (left) and 7 (right)
using Robotutils' camera intrinsics and Arlo drive.

- Uses your Robotutils.camera.Camera(width=1640, height=1232, fx=..., fy=..., cx, cy)
- Old/new OpenCV ArUco API compatible
- Adds 0.2 s settle after each spin to avoid motion blur
"""

import math, time, sys
from collections import namedtuple
import numpy as np
import cv2

# ---------------- Project imports ----------------
from Robotutils import robot as robot_mod
from Robotutils import CameraDetection_util as ru_camera  # <- your camera module (with Camera class)

# ---------------- Config ----------------
MARKER_ID_LEFT  = 6
MARKER_ID_RIGHT = 7
LANDMARK_BASELINE_M = 3.00     # meters between the two boxes
MARKER_SIZE_M        = 0.140   # marker edge (m)

# tolerances + motion (tune if needed)
CENTER_X_TOL_PX   = 20
EQUAL_DIST_TOL_M  = 0.06
R_TARGET_TOL_M    = 0.06
MAX_RUNTIME_S     = 240

TURN_POWER        = 50
TURN_SEC_PER_RAD  = 0.50
FWD_POWER         = 60
SEC_PER_M         = 2.40
SAFETY_MIN_FRONT_MM = 300

# ---------------- Intrinsics are taken from Robotutils.camera.Camera -----------
ArucoObs = namedtuple('ArucoObs', 'id x_px y_px h_px')

def make_camera():
    """
    Build your Robotutils camera with your defaults:
      width=1640, height=1232, fps=30, fx=1360, fy=1360, cx=width/2, cy=height/2
    If your Camera needs different args, adjust here.
    """
    cam = ru_camera.Camera(width=1640, height=1232, fx=1360, fy=1360, fps=30)
    # If your Camera has an explicit start() method, call it:
    if hasattr(cam, "start") and callable(cam.start):
        cam.start()
        time.sleep(0.2)
    return cam

def grab_frame(cam):
    """
    Compatible with multiple Robotutils camera variants:
      - ru_camera.grab_frame(cam)
      - cam.grab_frame()
      - cam.get_frame()
      - cam.read() -> (ok, frame)
      - cam.frame
    """
    # module-level helper
    if hasattr(ru_camera, "grab_frame") and callable(ru_camera.grab_frame):
        return ru_camera.grab_frame(cam)

    # object helpers
    for name in ("grab_frame", "get_frame"):
        if hasattr(cam, name) and callable(getattr(cam, name)):
            return getattr(cam, name)()
    if hasattr(cam, "read") and callable(cam.read):
        ok, frm = cam.read()
        if not ok:
            raise RuntimeError("Camera read() failed")
        return frm
    if hasattr(cam, "frame"):
        return cam.frame

    raise RuntimeError("No compatible frame-grab method found on Robotutils camera")

def detect_aruco(frame):
    """Detect ArUco markers (DICT_6X6_250), compatible with old/new OpenCV APIs."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ad = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    # Try new API first; fall back to old
    try:
        try:
            params = cv2.aruco.DetectorParameters()
        except Exception:
            params = cv2.aruco.DetectorParameters_create()
        detector = cv2.aruco.ArucoDetector(ad, params)
        corners, ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ad, parameters=params)

    out = []
    if ids is None:
        return out
    for i, c in enumerate(corners):
        pts = c[0]  # TL, TR, BR, BL
        xs, ys = pts[:, 0], pts[:, 1]
        x = float(xs.mean()); y = float(ys.mean())
        # vertical pixel size = average of left/right edges
        left_h  = math.hypot(pts[3,0]-pts[0,0], pts[3,1]-pts[0,1])
        right_h = math.hypot(pts[2,0]-pts[1,0], pts[2,1]-pts[1,1])
        h_px = 0.5 * (left_h + right_h)
        out.append(ArucoObs(int(ids[i][0]), x, y, h_px))
    return out

def obs_to_distance_bearing(obs: ArucoObs, fx, fy, cx, cy):
    """
    Distance Z = (fy * H) / h_px  (use fy because h_px is vertical)
    Bearing  φ = atan((x - cx) / fx)
    """
    if obs.h_px <= 1.0:
        return None
    d_m = (fy * MARKER_SIZE_M) / obs.h_px
    bearing_rad = math.atan((obs.x_px - cx) / fx)
    return d_m, bearing_rad

def pick_two_markers(obs_list):
    by_id = {o.id: o for o in obs_list}
    if MARKER_ID_LEFT in by_id and MARKER_ID_RIGHT in by_id:
        return by_id[MARKER_ID_LEFT], by_id[MARKER_ID_RIGHT]
    return None

# ---------------- Motion (with anti-blur pause) ----------------
def go_spin(arlo, radians):
    """Spin in place by 'radians'. Pause 0.2s after to avoid motion blur."""
    if abs(radians) < 1e-6:
        return
    sec = abs(radians) * TURN_SEC_PER_RAD
    left_dir, right_dir = (0,1) if radians < 0 else (1,0)  # CW if negative
    arlo.go_diff(TURN_POWER, TURN_POWER, left_dir, right_dir)
    time.sleep(sec)
    arlo.stop()
    time.sleep(0.2)  # settle camera

def go_forward(arlo, meters):
    if abs(meters) < 1e-6:
        return
    sec = abs(meters) * SEC_PER_M
    fwd = 1 if meters > 0 else 0
    arlo.go_diff(FWD_POWER, FWD_POWER, fwd, fwd)
    t0 = time.time()
    while time.time() - t0 < sec:
        try:
            mm = arlo.read_front_ping_sensor()
            if isinstance(mm, int) and 0 < mm < SAFETY_MIN_FRONT_MM:
                break
        except Exception:
            pass
        time.sleep(0.02)
    arlo.stop()
    time.sleep(0.2)  # small settle

# ---------------- Main control loop ----------------
def control_loop():
    # Robot + Camera
    arlo = robot_mod.Robot()
    cam = make_camera()

    # Pull intrinsics from YOUR camera object
    try:
        fx = float(cam.fx); fy = float(cam.fy)
        cx = float(cam.cx); cy = float(cam.cy)
        IMG_W = int(cam.width); IMG_H = int(cam.height)
    except Exception as e:
        print("ERROR: Could not read intrinsics from Robotutils camera:", e, file=sys.stderr)
        sys.exit(1)

    print(f"[Camera] {IMG_W}x{IMG_H}, fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    # 1) Spin until both markers visible
    print("Scanning for markers 6 and 7…")
    t0 = time.time()
    while True:
        frame = grab_frame(cam)
        obs = detect_aruco(frame)
        pair = pick_two_markers(obs)
        if pair:
            break
        go_spin(arlo, +0.15)  # ~8.6°
        if time.time() - t0 > MAX_RUNTIME_S:
            print("Timeout: could not see both markers.")
            return

    # 2) Drive to midpoint
    target_mean = LANDMARK_BASELINE_M / 2.0
    print("Both markers found. Driving to midpoint…")

    while True:
        frame = grab_frame(cam)
        obs = detect_aruco(frame)
        pair = pick_two_markers(obs)
        if not pair:
            go_spin(arlo, +0.10)
            continue

        left, right = pair
        m1 = obs_to_distance_bearing(left, fx, fy, cx, cy)
        m2 = obs_to_distance_bearing(right, fx, fy, cx, cy)
        if (m1 is None) or (m2 is None):
            go_spin(arlo, +0.05)
            continue

        d1, phi1 = m1
        d2, phi2 = m2

        # (a) face the midpoint in image coordinates
        avg_x = 0.5 * (left.x_px + right.x_px)
        dx = avg_x - cx
        if abs(dx) > CENTER_X_TOL_PX:
            go_spin(arlo, radians=-0.0025 * dx)  # proportional micro spin
            continue

        # (b) equalize distances => perpendicular bisector
        equal_err = d1 - d2
        if abs(equal_err) > EQUAL_DIST_TOL_M:
            sign = +1.0 if equal_err < 0 else -1.0
            go_spin(arlo, radians=0.05 * sign)
            continue

        # (c) move to radial target: mean(d1,d2) -> L/2
        mean_d = 0.5 * (d1 + d2)
        r_err = mean_d - target_mean
        if abs(r_err) > R_TARGET_TOL_M:
            go_forward(arlo, meters=r_err)
            continue

        print(f"Centered: d1={d1:.2f} m, d2={d2:.2f} m, mean≈{mean_d:.2f} m (target {target_mean:.2f} m).")
        break

    arlo.stop()
    time.sleep(0.2)

if __name__ == "__main__":
    try:
        control_loop()
    except KeyboardInterrupt:
        pass
