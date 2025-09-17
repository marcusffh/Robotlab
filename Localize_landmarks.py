import cv2
import cv2.aruco as aruco
import numpy as np
import time
from Exercise1.CalibratedRobot import CalibratedRobot

# Initialize robot
calArlo = CalibratedRobot()

# ================= CAMERA HELPERS =================
def make_camera(width=960, height=720, fps=30):
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        frame_dur = int(1.0/fps * 1_000_000)
        cfg = cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameDurationLimits": (frame_dur, frame_dur)},
            queue=False
        )
        cam.configure(cfg)
        cam.start()
        time.sleep(0.8)
        def read_fn():
            rgb = cam.capture_array("main")
            return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        def release_fn():
            try: cam.stop()
            except: pass
        return read_fn, release_fn, width, height
    except Exception as e:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError(f"Camera init failed: {e}")
        def read_fn():
            ok, frame = cap.read()
            return ok, frame
        def release_fn():
            try: cap.release()
            except: pass
        return read_fn, release_fn, width, height

read_fn, release_fn, width, height = make_camera()

# Camera parameters
focal_length = 1275  # pixels
camera_matrix = np.array([[focal_length, 0, width/2],
                          [0, focal_length, height/2],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))

# ArUco dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

# ================= SEARCH + DRIVE =================
def search_and_drive():
    marker_size = 140      # mm
    STOP_BUFFER = 0.3      # meters
    driving = False

    while True:
        ret, frame = read_fn()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None and len(ids) > 0:
            # Estimate pose
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, marker_size, camera_matrix, dist_coeffs
            )
            rvec = rvecs[0]
            tvec = tvecs[0][0]   # marker translation vector

            # Compute forward vector to marker in camera frame
            v = tvec / np.linalg.norm(tvec)      # unit vector to marker
            beta = np.arccos(np.clip(np.dot(v, [0,0,1]), -1.0, 1.0))  # angle to forward axis
            beta_deg = np.degrees(beta)

            # Determine sign: left or right
            if tvec[0] < 0:
                beta_deg = -beta_deg

            # Distance to marker (standoff)
            distance = max(np.linalg.norm(tvec)/1000.0 - STOP_BUFFER, 0)

            print(f"Detected marker IDs: {ids.flatten()}")
            print(f"tvec: {tvec}, angle: {beta_deg:.2f}°, distance: {distance:.2f} m")

            # Turn once toward marker
            calArlo.turn_angle(beta_deg)

            # Drive toward marker only if distance > 0
            if distance > 0 and not driving:
                calArlo.drive_distance(distance)
                driving = True

            if distance <= 0:
                print("Reached marker!")
                calArlo.stop()
                break
        else:
            print("Searching for marker...")
            calArlo.drive(25, 25, calArlo.BACKWARD, calArlo.FORWARD)
            time.sleep(0.2)
            calArlo.stop()
            driving = False

try:
    search_and_drive()
finally:
    release_fn()
    calArlo.stop()
    cv2.destroyAllWindows()