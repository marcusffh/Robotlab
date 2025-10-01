#!/usr/bin/env python3
"""
camera_noah.py
- Small wrapper that yields ArUco detections as: [(id, distance_m, bearing_rad), ...]
- Bearing is positive to the LEFT (standard image coords x increasing to the right).
- Distance is computed from marker pixel size using a pinhole model.
- Works headless over SSH. Falls back to a "mock" camera if Picamera2/OpenCV aren't available.

Tune:
  MARKER_SIZE_M: physical side length of your ArUco markers (meters)
  F_PX:          focal length in pixels (use your calibrated value)
  IMG_W/IMG_H:   capture size (keep moderate for speed on Pi)
"""

import math
import time

# --- Tunables (edit to match your setup) -------------------------------------
MARKER_SIZE_M = 0.140     # 140 mm square marker
F_PX          = 1275.0    # your calibrated focal length in pixels
IMG_W, IMG_H  = 960, 720  # capture resolution
FPS           = 15

# -----------------------------------------------------------------------------
try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from picamera2 import Picamera2
    _HAS_PICAM2 = True
except Exception:
    _HAS_PICAM2 = False


class LandmarkCamera:
    """
    Unified interface:
        cam = LandmarkCamera()
        cam.open()
        detections = cam.read()  # list[(id, dist_m, bearing_rad), ...]
        cam.close()
    """
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock or (not _HAS_CV2)
        self.cam = None
        self._last_mock_time = time.time()
        self._mock_angle = -0.6  # start with a bearing to simulate "spin to find"
        self._mock_seen_ids = set()

        # pre-init aruco
        if _HAS_CV2 and not self.use_mock:
            # Use 6x6_250 (change if your markers differ)
            self._aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            self._detpar = cv2.aruco.DetectorParameters()
            self._det = cv2.aruco.ArucoDetector(self._aruco, self._detpar)
            self._cx0, self._cy0 = IMG_W / 2.0, IMG_H / 2.0

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
            # Warm up
            time.sleep(0.5)
        else:
            # Fallback to cv2.VideoCapture(0) if USB cam
            self.cam = cv2.VideoCapture(0)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)
            self.cam.set(cv2.CAP_PROP_FPS, FPS)
            time.sleep(0.2)

    def close(self):
        if self.use_mock:
            return
        if _HAS_PICAM2 and self.cam is not None:
            self.cam.stop()
            self.cam = None
        elif self.cam is not None:
            try:
                self.cam.release()
            except Exception:
                pass
            self.cam = None

    def _grab_frame(self):
        if _HAS_PICAM2 and isinstance(self.cam, Picamera2):
            return self.cam.capture_array()
        else:
            ok, frame = self.cam.read()
            if not ok:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _detect_aruco(self, rgb):
        corners, ids, _ = self._det.detectMarkers(rgb)
        if ids is None or len(ids) == 0:
            return []

        dets = []
        for i, cid in enumerate(ids.flatten().tolist()):
            pts = corners[i][0]  # 4x2
            # pixel size from vertical extent (more stable on the Pi)
            y_coords = pts[:, 1]
            height_px = max(y_coords) - min(y_coords)
            if height_px <= 1.0:
                continue

            # distance from pinhole model: f = x*Z/X  =>  Z = f * X / x
            Z_m = F_PX * MARKER_SIZE_M / float(height_px)

            # bearing from principal point:
            cx = pts[:, 0].mean()
            bearing = math.atan2((cx - self._cx0), F_PX)  # small-angle approx safe

            dets.append((cid, Z_m, bearing))
        return dets

    def _mock_read(self):
        # Very simple mock: two IDs at +/- 0.35 rad @ ~2.5–3.2 m.
        now = time.time()
        dt = now - self._last_mock_time
        self._last_mock_time = now

        # Simulate slow spin
        self._mock_angle += 0.2 * dt
        out = []
        # ID 10 shows around angle ~ -0.2..0.2
        if -0.15 < self._mock_angle < 0.15:
            out.append((10, 2.6, self._mock_angle))
            self._mock_seen_ids.add(10)
        # ID 20 shows around angle ~ 0.8..1.1
        if 0.85 < self._mock_angle < 1.10:
            out.append((20, 3.0, self._mock_angle - 0.95))
            self._mock_seen_ids.add(20)
        if self._mock_angle > 1.2:
            self._mock_angle = -0.6
        return out

    def read(self):
        """Return list of (id, distance_m, bearing_rad)."""
        if self.use_mock:
            return self._mock_read()

        rgb = self._grab_frame()
        if rgb is None:
            return []
        try:
            return self._detect_aruco(rgb)
        except Exception:
            # If anything fails at runtime, don't crash the main loop.
            return []
