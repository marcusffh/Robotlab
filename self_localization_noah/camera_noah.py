# camera_noah.py — Picamera2 + OpenCV ArUco detector
# Returns [{'id': int, 'distance_cm': float, 'phi_rad': float}, ...]

import math
import cv2
import numpy as np

# ---- Camera & marker config (adjust to your setup) ----
MARKER_SIZE_CM = 14.0         # <-- side length of your printed markers (cm)
FOCAL_PX       = 1275.0       # <-- from your Ex2 calibration (pixels)
FRAME_W, FRAME_H = 960, 720   # modest for speed

# ---- OpenCV ArUco setup ----
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

if hasattr(cv2.aruco, "ArucoDetector"):
    # New API (OpenCV >= 4.7)
    DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())
    def detect_markers(gray):
        return DETECTOR.detectMarkers(gray)
else:
    # Legacy API (OpenCV <= 4.6)
    PARAMS = cv2.aruco.DetectorParameters_create()
    def detect_markers(gray):
        return cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=PARAMS)

# ---- Picamera2 setup ----
try:
    from picamera2 import Picamera2
except ImportError as e:
    raise RuntimeError(
        "Picamera2 missing. Install with: sudo apt install -y python3-picamera2 libcamera-apps"
    ) from e


def compress_duplicates_by_id(dets):
    """Keep nearest detection per ArUco ID (handles multiple faces)."""
    best = {}
    for d in dets:
        lid = int(d["id"])
        if lid not in best or d["distance_cm"] < best[lid]["distance_cm"]:
            best[lid] = d
    return list(best.values())


class CameraNoah:
    def __init__(self):
        self.cam = Picamera2()
        cfg = self.cam.create_video_configuration(
            main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
        )
        self.cam.configure(cfg)
        self.cam.start()
        self.cx = FRAME_W / 2.0  # principal point (px) — ok to use image center
        self.cy = FRAME_H / 2.0

    def close(self):
        try:
            self.cam.stop()
        except Exception:
            pass

    @staticmethod
    def _distance_from_px(px_len):
        # pinhole: Z = f * X / x
        if px_len <= 0:
            return None
        return (FOCAL_PX * MARKER_SIZE_CM) / px_len

    def _bearing_from_cx(self, cx_px):
        # tan(phi) = (u - cx)/f → phi = atan2(u-cx, f)
        return math.atan2((cx_px - self.cx), FOCAL_PX)

    def detect(self):
        """Return detections as list of dicts: {'id', 'distance_cm', 'phi_rad'}."""
        frame = self.cam.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        corners, ids, _ = detect_markers(gray)
        if ids is None or len(ids) == 0:
            return []

        ids = ids.flatten().astype(int)
        out = []
        for i, pts in zip(ids, corners):
            quad = pts.reshape(-1, 2)
            cx = float(np.mean(quad[:, 0]))

            # side length in pixels as avg of two edges
            e1 = np.linalg.norm(quad[0] - quad[1])
            e2 = np.linalg.norm(quad[1] - quad[2])
            px_len = float(0.5 * (e1 + e2))

            dist_cm = self._distance_from_px(px_len)
            if dist_cm is None:
                continue

            phi = self._bearing_from_cx(cx)

            out.append({
                "id": int(i),
                "distance_cm": float(dist_cm),
                "phi_rad": float(phi),
            })

        return compress_duplicates_by_id(out)
