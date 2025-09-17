import cv2
import cv2.aruco as aruco
import numpy as np
import time
from Exercise1.CalibratedRobot import CalibratedRobot



# Initialize robot
calArlo = CalibratedRobot()

# Camera parameters (replace fx, fy, cx, cy with your calibration results)
focal_length = 1275  # example from task 1
camera_matrix = np.array([[focal_length, 0, 960/2],
                          [0, focal_length, 720/2],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))  # assume no distortion if not calibrated

# ArUco dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()


# ================= CAMERA HELPERS =================
def make_camera(width=960, height=720, fps=30):
    """Return (read_fn, release_fn) that yields BGR frames."""
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
        def gst(w, h, f):
            return ("libcamerasrc ! videobox autocrop=true ! "
                    f"video/x-raw, width=(int){w}, height=(int){h}, framerate=(fraction){f}/1 ! "
                    "videoconvert ! appsink")
        cap = cv2.VideoCapture(gst(width, height, fps), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError(f"Camera init failed: {e}")
        def read_fn():
            ok, frame = cap.read()
            return ok, frame
        def release_fn():
            try: cap.release()
            except: pass
        return read_fn, release_fn
    
    
read_fn, release_fn = make_camera()
def search_and_drive():
    marker_size = 140   

    while True:
        ret, frame = read_fn()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            # Draw detected markers
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Pose estimation
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, marker_size, camera_matrix, dist_coeffs
            )
            tvec = tvecs[0][0]  # (x, y, z) in mm

            # Draw axis for first marker
            aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], marker_size/2)

            # Compute angle and distance
            angle = np.degrees(np.arctan2(tvec[0], tvec[2]))
            dist = tvec[2] / 1000.0  # convert mm → meters

            # Debug info
            print(f"Detected marker IDs: {ids.flatten()}")
            print(f"tvec: {tvec}, angle: {angle:.2f}°, distance: {dist:.2f} m")

            # Turn toward marker
            calArlo.turn_angle(angle)

            # Move forward (limit step to avoid overshooting)
            calArlo.drive_distance(min(dist, 0.3))

            # Stop if close enough
            if dist < 0.2:
                print("Reached landmark!")
                break
        else:
            print("Searching for marker...")
            calArlo.drive(50, 50, calArlo.BACKWARD, calArlo.FORWARD)
            time.sleep(0.2)
            calArlo.stop()

        # Show the camera feed
        cv2.imshow("Arlo Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


try:
    search_and_drive()
finally:
    release_fn()
    calArlo.stop()
    cv2.destroyAllWindows()