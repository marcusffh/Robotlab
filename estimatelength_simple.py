#!/usr/bin/env python3
"""
Simplified focal length estimation with ArUco / object size
-----------------------------------------------------------
Shows camera feed, detects ArUco marker, and prints
the estimated focal length f = x*Z/X for each capture.

Controls:
  - Press 'c' to capture and compute focal length
  - Press 'q' to quit
"""

import cv2
import numpy as np
import time

# --- Camera setup ---
def make_camera():
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        config = cam.create_video_configuration(main={"size": (640,480), "format":"RGB888"})
        cam.configure(config)
        cam.start()
        time.sleep(1)
        def read_fn():
            frame = cam.capture_array("main")
            return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return read_fn
    except Exception:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("No camera found!")
        def read_fn():
            return cap.read()
        return read_fn

# --- ArUco detection ---
def detect_marker(frame):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(frame, dictionary, parameters=parameters)
    if ids is None: return None
    pts = corners[0].reshape(-1,2)
    TL,TR,BR,BL = pts
    v1,v2 = np.linalg.norm(TL-BL), np.linalg.norm(TR-BR)
    x_px = 0.5*(v1+v2)
    return x_px, pts

def main():
    marker_mm = 80     # physical marker edge size [mm]
    Z = 500            # distance [mm] (adjust when testing)

    read_fn = make_camera()
    cv2.namedWindow("Camera")

    while True:
        ok, frame = read_fn()
        if not ok: break
        result = detect_marker(frame)
        if result:
            x_px, pts = result
            f_est = x_px * Z / marker_mm
            cv2.polylines(frame,[pts.astype(int)],True,(0,255,0),2)
            cv2.putText(frame,f"f={f_est:.1f}px",(10,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow("Camera",frame)

        key = cv2.waitKey(1) & 0xFF
        if key==ord('c') and result:
            print(f"Captured: Z={Z}mm, x={x_px:.1f}px -> f={f_est:.1f}px")
        if key==ord('q'):
            break

    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
