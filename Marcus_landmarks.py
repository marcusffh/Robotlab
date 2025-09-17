# find_landmark_search_then_drive_updated.py
import time
import numpy as np
from Robotutils.CameraDetection_util import CameraUtils, ArucoUtils
import Robotutils.robot as robot
import cv2
from Robotutils.CalibratedRobot import CalibratedRobot


IMG_W, IMG_H, FPS  = 960, 720, 30


# ==== Init Robot + Camera + Aruco ====
arlo = CalibratedRobot
cam = CameraUtils(width=IMG_W, height=IMG_H, fx=F_PX, fy=F_PX)
cam.start_camera(width=IMG_W, height=IMG_H, fps=FPS)
aruco = ArucoUtils()  # convert mm → m

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
