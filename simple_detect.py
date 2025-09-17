# map_landmarks_once.py — map all ArUco landmarks in front of the robot (2D: X,Z)
import time, cv2, numpy as np

# ---- Camera / marker params (set these) ----
F_PX         = 1275.0          # focal length [px] from Exercise 1
MARKER_SIZE_M= 0.140           # marker side length [m] (140 mm)
IMG_W, IMG_H = 960, 720        # capture size
FPS          = 30

# ---- Intrinsics (assume principal point at image center; zero distortion) ----
K = np.array([[F_PX,   0.0, IMG_W/2.0],
              [  0.0, F_PX, IMG_H/2.0],
              [  0.0,   0.0,        1.0]], dtype=np.float32)
dist = np.zeros((5,1), dtype=np.float32)

def make_camera(w=IMG_W, h=IMG_H, fps=FPS):
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        us = int(1.0/fps * 1_000_000)
        cfg = cam.create_video_configuration(main={"size": (w,h), "format":"RGB888"},
                                             controls={"FrameDurationLimits": (us,us)},
                                             queue=False)
        cam.configure(cfg); cam.start(); time.sleep(0.8)
        def read():  # returns BGR
            rgb = cam.capture_array("main")
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
            print("No frame"); return []

        aruco = cv2.aruco
        DICT  = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)  # your 6x6 tags
        params= aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(frame, DICT, parameters=params)
        if ids is None or len(corners)==0:
            print("No markers seen."); return []

        # Pose for all markers (units follow MARKER_SIZE_M → meters)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_M, K, dist)

        # Build simple (id, X, Z) list in meters; camera frame: +Z forward, +X right
        out = []
        for i, mid in enumerate(ids.flatten()):
            t = tvecs[i][0]  # (x,y,z) in meters
            out.append((int(mid), float(t[0]), float(t[2])))

        # Pretty print
        print("\nID    X_right[m]   Z_forward[m]")
        for mid, X, Z in out:
            print(f"{mid:3d}   {X:+8.3f}     {Z:+8.3f}")

        # Optional quick plot (comment out if running headless)
        try:
            import matplotlib.pyplot as plt
            xs = [X for _,X,_ in out]
            zs = [Z for *_,Z in out]
            plt.figure(); plt.scatter(xs, zs)
            for mid,X,Z in out:
                plt.text(X, Z, str(mid), fontsize=9, ha='left', va='bottom')
            plt.gca().invert_yaxis()  # optional if you prefer Z increasing downwards
            plt.xlabel("X (right, m)"); plt.ylabel("Z (forward, m)")
            plt.title("Landmarks in camera frame"); plt.axis('equal'); plt.grid(True)
            plt.show()
        except Exception:
            pass

        return out
    finally:
        try: release()
        except: pass

if __name__ == "__main__":
    map_landmarks_once()
