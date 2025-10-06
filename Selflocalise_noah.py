#!/usr/bin/env python3
"""
Self-localize (two known ArUco markers) and drive to their midpoint.

Strategy (simple + robust):
1) Spin in place until both landmarks (by ID) appear in the same frame.
2) For every frame:
   - Measure distance d [m] to each marker from its pixel height:  Z = f_px * H / h_px
   - Measure bearing φ [rad] for each from its horizontal pixel offset: φ = atan((x - cx)/fx)
   - Face the midpoint on screen (make the average x of the two markers ≈ cx) by small turns.
   - Drive forward/backward until:
        a) distances are ~equal (on the perpendicular bisector), and
        b) their mean distance ≈ L/2 (you are at the center).
3) Stop.

Tune ONLY the CAL_* constants below for your robot/camera.
"""

import math, time, sys
from collections import namedtuple

import numpy as np
import cv2

# If Picamera2 is available (recommended on the Pi)
try:
    from picamera2 import Picamera2
    USE_PICAM2 = True
except ImportError:
    USE_PICAM2 = False

# --- Arlo robot API ----------------------------------------------------------
from Robotutils import robot  # your provided API (go_diff, stop, sonar, etc.)
# Robot doc/API reference: see robot.py. :contentReference[oaicite:0]{index=0}

# =============================================================================
# USER CALIBRATION (EDIT THESE)
# =============================================================================
# Landmark IDs and world layout (Exercise 5 default: markers at (0,0) and (3.0, 0) meters)
# Put the *printed* ArUco IDs here to bind them to the known 1D layout.
MARKER_ID_LEFT  = 6   # TODO: set to your "left" box ID (as seen from the robot start area)
MARKER_ID_RIGHT = 7   # TODO: set to your "right" box ID
LANDMARK_BASELINE_M = 3.00  # 3.0 m apart per handout (300 cm). :contentReference[oaicite:1]{index=1}

# ArUco marker physical size (edge length of one square on the box)
MARKER_SIZE_M = 0.140  # 14 cm from Ex. 2/handout defaults. Adjust if you use a different print.

# Camera intrinsics (fx≈fy=f_px) — quick pinhole model
# If you have a better calibration, put it here (fx, fy, cx, cy). Otherwise these work decently.
IMG_W, IMG_H = 1280, 720
F_PX = 1360.0  # focal length in pixels from your earlier estimation; tweak ±10% if needed. :contentReference[oaicite:2]{index=2}
CX, CY = IMG_W/2.0, IMG_H/2.0

# Motion calibration (time-based, no encoders)
TURN_POWER = 50         # 30–90 ok
TURN_SEC_PER_RAD = 0.50 # seconds of spin for 1 rad (≈28.6°). TUNE on your floor!
FWD_POWER = 60
SEC_PER_M = 2.40        # seconds to drive 1 meter straight. TUNE.
SAFETY_MIN_FRONT_MM = 300  # sonar stop distance

# Control tolerances
CENTER_X_TOL_PX = 20                  # keep midpoint of markers near image center
EQUAL_DIST_TOL_M = 0.06               # |d1 - d2| <= tol → on perpendicular bisector
R_TARGET_TOL_M = 0.06                 # |(d1+d2)/2 - L/2| <= tol → at radial target
MAX_RUNTIME_S = 240                   # failsafe

# =============================================================================

ArucoObs = namedtuple('ArucoObs', 'id x_px y_px h_px')

def make_camera():
    if USE_PICAM2:
        cam = Picamera2()
        cfg = cam.create_video_configuration(
            main={"size": (IMG_W, IMG_H), "format": "RGB888"},
            controls={}
        )
        cam.configure(cfg)
        cam.start()
        time.sleep(0.5)
        return cam
    else:
        # gstreamer path (works with libcamera via OpenCV on many images)
        gst = (
            "libcamerasrc ! "
            "videobox autocrop=true ! "
            f"video/x-raw, width=(int){IMG_W}, height=(int){IMG_H}, framerate=(fraction)30/1 ! "
            "videoconvert ! appsink"
        )
        cap = cv2.VideoCapture(gst, apiPreference=cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print("ERROR: Could not open camera.", file=sys.stderr)
            sys.exit(1)
        return cap

def grab_frame(cam):
    if USE_PICAM2:
        return cam.capture_array("main")
    else:
        ok, frame = cam.read()
        if not ok:
            raise RuntimeError("Camera frame read failed")
        return frame

def detect_aruco(frame):
    """Return list of ArucoObs for all detected markers (DICT_6X6_250)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ad = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(ad, params)
    corners, ids, _ = detector.detectMarkers(gray)
    out = []
    if ids is None: 
        return out
    for i, c in enumerate(corners):
        pts = c[0]  # (4,2): TL, TR, BR, BL
        xs = pts[:,0]; ys = pts[:,1]
        x = float(xs.mean()); y = float(ys.mean())
        # pixel height = average of left/right edge lengths
        left_h  = math.hypot(pts[3,0]-pts[0,0], pts[3,1]-pts[0,1])
        right_h = math.hypot(pts[2,0]-pts[1,0], pts[2,1]-pts[1,1])
        h_px = 0.5*(left_h + right_h)
        out.append(ArucoObs(int(ids[i][0]), x, y, h_px))
    return out

def obs_to_distance_bearing(obs: ArucoObs):
    """Pinhole projection: Z = f * H / h_px; bearing ≈ atan((x-cx)/fx)."""
    if obs.h_px <= 1.0:
        return None
    d_m = (F_PX * MARKER_SIZE_M) / obs.h_px
    bearing_rad = math.atan((obs.x_px - CX) / F_PX)
    return d_m, bearing_rad

def go_spin(arlo, radians):
    """Positive = left (CCW), negative = right (CW). Time-based."""
    if radians == 0.0: 
        return
    sec = abs(radians) * TURN_SEC_PER_RAD
    left_dir, right_dir = (0,1) if radians < 0 else (1,0)  # CW vs CCW
    arlo.go_diff(TURN_POWER, TURN_POWER, left_dir, right_dir)
    time.sleep(sec)
    arlo.stop(); time.sleep(0.05)

def go_forward(arlo, meters):
    if meters == 0.0: 
        return
    sec = abs(meters) * SEC_PER_M
    fwd = 1 if meters > 0 else 0
    arlo.go_diff(FWD_POWER, FWD_POWER, fwd, fwd)
    t0 = time.time()
    while time.time() - t0 < sec:
        # Safety stop
        try:
            front_mm = arlo.read_front_ping_sensor()
            if isinstance(front_mm, int) and 0 < front_mm < SAFETY_MIN_FRONT_MM:
                break
        except Exception:
            pass
        time.sleep(0.02)
    arlo.stop(); time.sleep(0.05)

def pick_two_markers(obs_list):
    """Return observations for (leftID, rightID) if both present, else None."""
    by_id = {o.id:o for o in obs_list}
    if MARKER_ID_LEFT in by_id and MARKER_ID_RIGHT in by_id:
        return by_id[MARKER_ID_LEFT], by_id[MARKER_ID_RIGHT]
    return None

def control_loop():
    arlo = robot.Robot()
    cam = make_camera()

    # quick helpers
    def frame_obs():
        frame = grab_frame(cam)
        obs = detect_aruco(frame)
        return frame, obs

    # 1) Spin until both IDs are visible
    print("Spinning to find both landmarks…")
    start = time.time()
    while True:
        frame, obs = frame_obs()
        pair = pick_two_markers(obs)
        if pair is not None:
            break
        go_spin(arlo, radians=+0.15)  # ~8.6° per step
        if time.time() - start > MAX_RUNTIME_S:
            print("Timeout: could not see both markers.")
            return

    # 2) Drive to midpoint
    L = LANDMARK_BASELINE_M
    target_mean = L/2.0
    print("Both markers found. Centering…")
    while True:
        # re-read until we see both again (spin small if needed)
        _, obs = frame_obs()
        pair = pick_two_markers(obs)
        if pair is None:
            # slow scan micro-steps to recover view
            go_spin(arlo, radians=+0.10)
            continue

        left, right = pair
        m1 = obs_to_distance_bearing(left)
        m2 = obs_to_distance_bearing(right)
        if m1 is None or m2 is None:
            go_spin(arlo, radians=+0.05)
            continue
        d1, phi1 = m1
        d2, phi2 = m2

        # (a) face midpoint in image ⇒ average of x’s near CX
        avg_x = 0.5*(left.x_px + right.x_px)
        dx = avg_x - CX
        if abs(dx) > CENTER_X_TOL_PX:
            # proportional tiny spin
            go_spin(arlo, radians=-0.0025 * dx)  # negative dx -> spin right
            continue

        # (b) equalize distances ⇒ on perpendicular bisector
        equal_err = d1 - d2
        if abs(equal_err) > EQUAL_DIST_TOL_M:
            # Nudge heading slightly toward the farther marker to equalize
            # If d1>d2 the robot is closer to LEFT; rotate slightly left so forward motion equalizes.
            sign = +1.0 if equal_err < 0 else -1.0
            go_spin(arlo, radians=0.05 * sign)
            continue

        # (c) move to radial target: (d1+d2)/2 -> L/2
        mean_d = 0.5*(d1 + d2)
        r_err = mean_d - target_mean
        if abs(r_err) > R_TARGET_TOL_M:
            go_forward(arlo, meters=r_err)  # positive -> too far: move forward
            continue

        # Success!
        print("Arrived at midpoint: d1=%.2f m, d2=%.2f m, mean=%.2f m ≈ L/2=%.2f m" %
              (d1, d2, mean_d, target_mean))
        break

    # full stop
    arlo.stop()
    time.sleep(0.2)

if __name__ == "__main__":
    try:
        control_loop()
    except KeyboardInterrupt:
        pass
