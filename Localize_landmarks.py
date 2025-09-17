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
    marker_size = 140   # mm

    while True:
        ret, frame = read_fn()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None and len(ids) > 0:
            # Pose estimation
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, marker_size, camera_matrix, dist_coeffs
            )
            tvec = tvecs[0][0]

            # Compute angle and distance
            angle = -np.degrees(np.arctan2(tvec[0], tvec[2]))  # flip sign if needed
            dist = tvec[2]/1000 
            dist = max(dist, 0)  # avoid negative distance

            print(f"Detected marker IDs: {ids.flatten()}")
            print(f"tvec: {tvec}, angle: {angle:.2f}°, distance: {dist:.2f} m")

            # Turn and move
            calArlo.turn_angle(angle)
            calArlo.drive_distance(dist)

            if dist <= 0:
                print("Reached landmark!")
                break
        else:
            print("Searching for marker...")
            calArlo.drive(50, 50, calArlo.BACKWARD, calArlo.FORWARD)
            time.sleep(0.2)
            calArlo.stop()

try:
    search_and_drive()
finally:
    release_fn()
    calArlo.stop()
    cv2.destroyAllWindows()
