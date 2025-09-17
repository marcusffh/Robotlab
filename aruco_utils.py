#!/usr/bin/env python3
"""
arucoutils.py
-------------
Minimal utilities you can reuse across scripts.

- Camera setup (Picamera2 preferred, else libcamera via GStreamer/OpenCV).
- ArUco (DICT_6X6_250) detection.
- Optional pose via estimatePoseSingleMarkers (if intrinsics + marker_size_m are provided).
- Simple helpers for selecting a target and computing heading error.
- Tiny wrappers that CALL your robot's public API (no redefinitions).

Dependencies: OpenCV with aruco contrib.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
import time
import numpy as np
import cv2


# ---------------- Camera / Intrinsics ----------------

@dataclass
class CameraConfig:
    width: int = 1640
    height: int = 1232
    fps: int = 30


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray  # shape (k,) or (k,1)


# ---------------- Detection container ----------------

@dataclass
class Detection:
    marker_id: int
    corners: np.ndarray               # (4,2) float32
    center_xy: Tuple[float, float]    # (cx, cy) pixels
    side_px: float                    # mean edge length (pixels)
    rvec: Optional[np.ndarray] = None # (3,)
    tvec: Optional[np.ndarray] = None # (3,)


# ---------------- Main utility class ----------------

class ArucoUtils:
    def __init__(
        self,
        cam_cfg: CameraConfig = CameraConfig(),
        intrinsics: Optional[Intrinsics] = None,
        marker_size_m: Optional[float] = None,          # physical side length (meters)
        aruco_dict_id: int = cv2.aruco.DICT_6X6_250,
    ):
        self.cam_cfg = cam_cfg
        self.intrinsics = intrinsics
        self.marker_size_m = marker_size_m

        self._dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self._params = cv2.aruco.DetectorParameters_create()

        self._picam2 = None
        self._cap = None
        self._started = False

    # -------- Camera --------

    def start_camera(self) -> None:
        """Start camera (Picamera2 preferred, else libcamera via GStreamer/OpenCV)."""
        if self._started:
            return
        try:
            from picamera2 import Picamera2
            self._picam2 = Picamera2()
            frame_dur = int(1.0 / self.cam_cfg.fps * 1_000_000)
            cfg = self._picam2.create_video_configuration(
                main={"size": (self.cam_cfg.width, self.cam_cfg.height), "format": "RGB888"},
                controls={"FrameDurationLimits": (frame_dur, frame_dur)},
                queue=False,
            )
            self._picam2.configure(cfg)
            self._picam2.start()
            time.sleep(1.0)
            self._started = True
            return
        except Exception:
            self._picam2 = None  # fall through to OpenCV/GStreamer

        gst = (
            "libcamerasrc ! videobox autocrop=true ! "
            f"video/x-raw, width=(int){self.cam_cfg.width}, height=(int){self.cam_cfg.height}, "
            f"framerate=(fraction){self.cam_cfg.fps}/1 ! "
            "videoconvert ! appsink"
        )
        self._cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise RuntimeError("Camera init failed (Picamera2 and GStreamer both unavailable).")
        self._started = True

    def read(self):
        """Return (ok, frame_bgr)."""
        if not self._started:
            self.start_camera()
        if self._picam2 is not None:
            rgb = self._picam2.capture_array("main")
            return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return self._cap.read()

    def stop_camera(self):
        if self._picam2 is not None:
            try: self._picam2.stop()
            except: pass
            self._picam2 = None
        if self._cap is not None:
            try: self._cap.release()
            except: pass
            self._cap = None
        self._started = False

    # -------- Detection / Pose --------

    @staticmethod
    def _mean_side_px(c4x2: np.ndarray) -> float:
        p = c4x2.reshape(4, 2)
        edges = [
            np.linalg.norm(p[1] - p[0]),
            np.linalg.norm(p[2] - p[1]),
            np.linalg.norm(p[3] - p[2]),
            np.linalg.norm(p[0] - p[3]),
        ]
        return float(np.mean(edges))

    @staticmethod
    def _center_xy(c4x2: np.ndarray) -> Tuple[float, float]:
        c = np.mean(c4x2.reshape(4, 2), axis=0)
        return float(c[0]), float(c[1])

    def detect(self, frame_bgr, restrict_ids: Optional[List[int]] = None, want_pose: Optional[bool] = None) -> List[Detection]:
        """
        Detect ArUco codes. If intrinsics + marker_size_m are set (or want_pose=True),
        also estimate pose with cv2.aruco.estimatePoseSingleMarkers.
        """
        if want_pose is None:
            want_pose = (self.intrinsics is not None and self.marker_size_m is not None)

        corners, ids, _ = cv2.aruco.detectMarkers(frame_bgr, self._dict, parameters=self._params)
        if ids is None or len(corners) == 0:
            return []

        rvecs = tvecs = None
        if want_pose:
            K = np.array([[self.intrinsics.fx, 0, self.intrinsics.cx],
                          [0, self.intrinsics.fy, self.intrinsics.cy],
                          [0, 0, 1]], dtype=np.float32)
            dist = self.intrinsics.dist.reshape(-1, 1).astype(np.float32)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, float(self.marker_size_m), K, dist)

        dets: List[Detection] = []
        for i, c in enumerate(corners):
            mid = int(ids[i][0])
            if restrict_ids is not None and mid not in restrict_ids:
                continue
            p = c.reshape(4, 2).astype(np.float32)
            dets.append(
                Detection(
                    marker_id=mid,
                    corners=p,
                    center_xy=self._center_xy(p),
                    side_px=self._mean_side_px(p),
                    rvec=(rvecs[i].reshape(3) if rvecs is not None else None),
                    tvec=(tvecs[i].reshape(3) if tvecs is not None else None),
                )
            )
        return dets

    @staticmethod
    def choose_largest(dets: List[Detection]) -> Optional[Detection]:
        """Pick the detection with the largest apparent size (closest)."""
        return max(dets, key=lambda d: d.side_px) if dets else None

    @staticmethod
    def yaw_from_tvec(tvec: np.ndarray) -> float:
        """Yaw error (rad): + if marker is to the right of camera forward axis."""
        x, _, z = float(tvec[0]), float(tvec[1]), float(tvec[2])
        return float(np.arctan2(x, z))

    # -------- Robot call-through helpers (no redefinitions) --------

    @staticmethod
    def rotate_step(bot, angle_deg: float, speed: Optional[int] = None) -> None:
        """Rotate the robot by a small step using CalibratedRobot.turn_angle()."""
        bot.turn_angle(angle_deg, speed=speed)

    @staticmethod
    def go_forward_step(bot, meters: float, speed: Optional[int] = None) -> None:
        """Move the robot forward a short distance using CalibratedRobot.drive_distance()."""
        bot.drive_distance(meters, direction=bot.FORWARD, speed=speed)
