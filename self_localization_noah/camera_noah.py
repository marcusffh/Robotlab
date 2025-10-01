#!/usr/bin/env python3
"""
camera_noah.py
- Yields ArUco detections as: [(id, distance_m, bearing_rad), ...]
- Bearing: positive to the LEFT (image x right; we compute angle around +y axis).
- Distance from pinhole model using marker pixel height.

Works headless (SSH). If OpenCV ArUco isn't available or you want to test on
a laptop: LandmarkCamera(use_mock=True).
"""

import math
import time

# --- Tunables (edit to match your setup) -------------------------------------
MARKER_SIZE_M = 0.140     # 140 mm
F_PX          = 1275.0    # your calibrated focal length in pixels
IMG_W, IMG_H  = 960, 720  # capture resolution
FPS           = 15
ARUCO_DICT_ID = "DICT_6X6_250"  # change if your markers differ
# -----------------------------------------------------------------------------

# Soft imports (allow running without camera/OpenCV)
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

try:
    import numpy as np
except Exception:
    # minimal fallback
    import array as np  # won't actually be used if CV2 missing
    np = None

try:
    from picamera2 import Picamera2
    _HAS_PICAM2 = True
except Exception:
    _HAS_PICAM2 = False


def _get_aruco_handles():
    """
    Return (has_aruco, dict_obj, det_params, new_api_detector_or_None).
    Supports:
      - Old API:  cv2.aruco.detectMarkers(image, dict, parameters=params)
      - New API:  cv2.aruco.ArucoDetector(dict, params).detectMarkers(image)
    """
    if not _HAS_CV2 or not hasattr(cv2, "aruco"):
        return False, None, None, None

    ar = cv2.aruco
    # dictionary getter (old vs new)
    if hasattr(ar, "getPredefinedDictionary"):
        dict_obj = getattr(ar, "getPredefinedDictionary")(getattr(ar, ARUCO_DICT_ID))
    else:
        dict_obj = getattr(ar, "Dictionary_get")(getattr(ar, ARUCO_DICT_ID))

    # parameters (old vs new)
    if hasattr(ar, "DetectorParameters_create"):
        det_params = ar.DetectorParameters_create()
    else:
        # Newer OpenCV may expose a class instead
        try:
            det_params = ar.DetectorParameters()
        except Exception:
            det_params = None

    # Detector object (new API in >= 4.7)
    new_api_detector = None
    if hasattr(ar, "ArucoDetector"):
        try:
            new_api_detector = ar.ArucoDetector(dict_obj, det_params)
        except Exception:
            new_api_detector = None

    return True, dict_obj, det_params, new_api_detector


class LandmarkCamera:
    """
    Unified interface:
        cam = LandmarkCamera(use_mock=False)
        cam.open()
        dets = cam.read()  # list of (id, dist_m, bearing_rad)
        cam.close()
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock or (not _HAS_CV2)
        self.cam = None
        self._last_mock_time = time.time()
        self._mock_angle = -0.6
        self._mock_seen_ids = set()

        self._cx0, self._cy0 = IMG_W / 2.0, IMG_H / 2.0

        # ArUco setup (if available)
        self._has_aruco, self._aruco_dict, self._aruco_params, self._aruco_detector = _get_aruco_handles()

        # If cv2 exists but aruco isn't compiled, force mock unless caller insists
        if _HAS_CV2 and not self._has_aruco and not self.use_mock:
            print("[camera_noah] WARNING: OpenCV built without aruco module. Using mock detections.")
            self.use_mock = True

    def open(self):
        if self.use_mock:
            return
        if _HAS_PICAM2:
            self.cam = Picamera2()
            us = int(1.0 / FPS * 1_000_000)
            cfg = self.cam.create_video_configuration(
                main={"size": (IMG_W, IMG_H), "format": "RGB888"},
                controls={"FrameDurationLimits": (us, us)},
                buffer_count=2
            )
            self.cam.configure(cfg)
            self.cam.start()
            time.sleep(0.5)
        else:
            # Fallback: USB cam via VideoCapture
            self.cam = cv2.VideoCapture(0)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)
            self.cam.set(cv2.CAP_PROP_FPS, FPS)
            time.sleep(0.2)

    def close(self):
        if self.use_mock:
            return
        if _HAS_PICAM2 and isinstance(self.cam, Picamera2):
            try:
                self.cam.stop()
            except Exception:
                pass
            self.cam = None
        elif self.cam is not None:
            try:
                self.cam.release()
            except Exception:
                pass
            self.cam = None

    def _grab_frame(self):
        if _HAS_PICAM2 and isinstance(self.cam, Picamera2):
            # Picamera2 returns RGB ndarray
            return self.cam.capture_array()
        else:
            ok, frame = self.cam.read()
            if not ok:
                return None
            # Convert BGR -> RGB
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _detect_aruco(self, rgb):
        """
        Return detections as [(id, Z_m, bearing_rad), ...]
        Works with both old and new OpenCV APIs.
        """
        if not self._has_aruco:
            return []

        if self._aruco_detector is not None:
            # New API (>=4.7)
            corners, ids, _ = self._aruco_detector.detectMarkers(rgb)
        else:
            # Old API
            try:
                corners, ids, _ = cv2.aruco.detectMarkers(rgb, self._aruco_dict, parameters=self._aruco_params)
            except Exception:
                corners, ids = None, None

        if ids is None or len(ids) == 0:
            return []

        dets = []
        for i, cid in enumerate(ids.flatten().tolist()):
            pts = corners[i][0]  # shape (4, 2)
            # vertical size (more stable)
            y_coords = pts[:, 1]
            height_px = float(max(y_coords) - min(y_coords))
            if height_px <= 1.0:
                continue

            # Pinhole model: f = x*Z/X  =>  Z = f*X/x
            Z_m = F_PX * MARKER_SIZE_M / height_px

            # Bearing from principal point (small-angle ok): atan((cx-cx0)/f)
            cx = float(pts[:, 0].mean())
            bearing = math.atan2((cx - self._cx0), F_PX)

            dets.append((cid, Z_m, bearing))

        return dets

    # ----------------------------- MOCK --------------------------------------
    def _mock_read(self):
        now = time.time()
        dt = now - self._last_mock_time
        self._last_mock_time = now

        # Simulate a slow spin revealing two markers at different angles
        self._mock_angle += 0.25 * dt
        out = []
        if -0.12 < self._mock_angle < 0.18:
            out.append((10, 2.6, self._mock_angle))
            self._mock_seen_ids.add(10)
        if 0.80 < self._mock_angle < 1.10:
            out.append((20, 3.0, self._mock_angle - 0.95))
            self._mock_seen_ids.add(20)
        if self._mock_angle > 1.2:
            self._mock_angle = -0.6
        return out

    # ------------------------------ API --------------------------------------
    def read(self):
        """Return list of (id, distance_m, bearing_rad)."""
        if self.use_mock:
            return self._mock_read()

        rgb = self._grab_frame()
        if rgb is None:
            return []
        try:
            return self._detect_aruco(rgb)
        except Exception as e:
            # Don't crash PF loop on sporadic CV errors
            # print(f"[camera_noah] detect error: {e}")
            return []
