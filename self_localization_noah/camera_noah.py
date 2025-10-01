#!/usr/bin/env python3
"""
camera_noah.py
- Wrapper for PiCam + ArUco detections.
- Returns list of (id, dist_m, bearing_rad).
- Provides draw_detections() for showing a frame with landmarks overlaid.
"""

import math
import time

MARKER_SIZE_M = 0.140     # 140 mm
F_PX          = 1275.0    # calibrated focal length (pixels)
IMG_W, IMG_H  = 960, 720
FPS           = 15
ARUCO_DICT_ID = "DICT_6X6_250"

# Soft imports
try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except Exception:
    cv2, np = None, None
    _HAS_CV2 = False

try:
    from picamera2 import Picamera2
    _HAS_PICAM2 = True
except Exception:
    _HAS_PICAM2 = False


def _get_aruco_handles():
    if not _HAS_CV2 or not hasattr(cv2, "aruco"):
        return False, None, None, None

    ar = cv2.aruco
    if hasattr(ar, "getPredefinedDictionary"):
        dict_obj = getattr(ar, "getPredefinedDictionary")(getattr(ar, ARUCO_DICT_ID))
    else:
        dict_obj = getattr(ar, "Dictionary_get")(getattr(ar, ARUCO_DICT_ID))

    if hasattr(ar, "DetectorParameters_create"):
        det_params = ar.DetectorParameters_create()
    else:
        det_params = ar.DetectorParameters()

    new_api_detector = None
    if hasattr(ar, "ArucoDetector"):
        try:
            new_api_detector = ar.ArucoDetector(dict_obj, det_params)
        except Exception:
            new_api_detector = None

    return True, dict_obj, det_params, new_api_detector


class LandmarkCamera:
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock or (not _HAS_CV2)
        self.cam = None
        self._cx0, self._cy0 = IMG_W/2, IMG_H/2
        self._has_aruco, self._aruco_dict, self._aruco_params, self._aruco_detector = _get_aruco_handles()
        if _HAS_CV2 and not self._has_aruco:
            print("[camera_noah] WARNING: OpenCV built without aruco module. Using mock.")
            self.use_mock = True

    def open(self):
        if self.use_mock: return
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
            self.cam = cv2.VideoCapture(0)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)
            self.cam.set(cv2.CAP_PROP_FPS, FPS)
            time.sleep(0.2)

    def close(self):
        if self.use_mock: return
        if _HAS_PICAM2 and isinstance(self.cam, Picamera2):
            try: self.cam.stop()
            except: pass
            self.cam = None
        elif self.cam is not None:
            try: self.cam.release()
            except: pass
            self.cam = None

    def _grab_frame(self):
        if self.use_mock: return None
        if _HAS_PICAM2 and isinstance(self.cam, Picamera2):
            return self.cam.capture_array()
        else:
            ok, frame = self.cam.read()
            if not ok: return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _detect_aruco(self, rgb):
        if not self._has_aruco: return []
        if self._aruco_detector is not None:
            corners, ids, _ = self._aruco_detector.detectMarkers(rgb)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(rgb, self._aruco_dict, parameters=self._aruco_params)
        if ids is None: return []
        dets = []
        for i, cid in enumerate(ids.flatten().tolist()):
            pts = corners[i][0]
            y_coords = pts[:,1]
            height_px = float(max(y_coords) - min(y_coords))
            if height_px <= 1.0: continue
            Z_m = F_PX * MARKER_SIZE_M / height_px
            cx = float(pts[:,0].mean())
            bearing = math.atan2((cx - self._cx0), F_PX)
            dets.append((cid, Z_m, bearing))
        return dets

    def read(self):
        if self.use_mock:
            return []
        rgb = self._grab_frame()
        if rgb is None: return []
        return self._detect_aruco(rgb)


def draw_detections(rgb, detections):
    """Overlay simple detection info on an RGB frame."""
    if rgb is None or not _HAS_CV2: return rgb
    out = rgb.copy()
    h, w = out.shape[:2]
    cx0, cy0 = w//2, h//2
    for (lid, dist, bearing) in detections:
        dx = int(math.tan(bearing) * (h//2))
        cv2.line(out, (cx0, cy0), (cx0+dx, cy0), (0,255,0), 2)
        cv2.putText(out, f"ID{lid} d={dist:.2f}m", (10,30+20*(lid%10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)  # back to BGR for imshow
