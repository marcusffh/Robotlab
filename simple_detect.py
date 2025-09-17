
import time, cv2, numpy as np
from picamera2 import Picamera2

# marker parameters
F_PX          = 1360.1         # focal length [px] from calibration
MARKER_SIZE_M = 0.140           # marker side length [m]
IMG_W, IMG_H  = 1640, 1232
FPS           = 30

# Intrinstic camera matrix
K = np.array([[F_PX, 0.0, IMG_W/2.0],
              [0.0,  F_PX, IMG_H/2.0],
              [0.0,  0.0,        1.0]], dtype=np.float32)
dist = np.zeros((5,1), dtype=np.float32)

def make_camera(w=IMG_W, h=IMG_H, fps=FPS):
    cam = Picamera2()
    us = int(1.0/fps * 1_000_000)
    cfg = cam.create_video_configuration(
        main={"size": (w, h), "format": "RGB888"},
        controls={"FrameDurationLimits": (us, us)},
        queue=False
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(0.8)
    def read():
        rgb = cam.capture_array("main")
        if rgb is None or rgb.size == 0:
            return False, None
        return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    def release():
        try: cam.stop()
        except: pass
    return read, release

def map_landmarks_once():
    read, release = make_camera()
    try:
        ok, frame = read()
        if not ok:
            raise RuntimeError("Failed to grab frame from Picamera2")

        aruco = cv2.aruco
        DICT  = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        params = aruco.DetectorParameters_create()

        corners, ids, _ = aruco.detectMarkers(frame, DICT, parameters=params)
        if ids is None or len(corners) == 0:
            print("No markers seen."); return []

        # Pose estimate, ignore rvec ('_' placeholder)
        _, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_M, K, dist)

        # Build (id, X, Z) list in meters, +Z forward, +X right
        out = []
        for i, mid in enumerate(ids.flatten()):
            t = tvecs[i][0]  # (x, y, z) in meters
            out.append((int(mid), float(t[0]), float(t[2])))

        # Print table
        print("\nID    X_right[m]   Z_forward[m]")
        for mid, X, Z in out:
            print(f"{mid:3d}   {X:+8.3f}     {Z:+8.3f}")

        return out
    finally:
        try: release()
        except: pass

if __name__ == "__main__":
    map_landmarks_once()
