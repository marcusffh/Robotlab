#!/usr/bin/env python3
# Picamera2-only ArUco utilities (no grayscale preproc; DICT_6X6_250).
# Detection logic mirrors the version that worked for you.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
import time
import numpy as np
import cv2
from picamera2 import Picamera2  # hard requirement

# ---------- Small data holders ----------

@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray  # shape (k,) or (k,1)

@dataclass
class Detection:
    marker_id: int
    corners: np.ndarray               # (4,2) float32: TL, TR, BR, BL
    center_xy: Tuple[float, float]    # (cx, cy) pixels
    side_px: float                    # mean side length (px) — vertical sides average
    rvec: Optional[np.ndarray] = None # (3,) pose if intrinsics+size are provided
    tvec: Optional[np.ndarray] = None # (3,)

# ---------- Main utility ----------

class ArucoUtils:
    def __init__(
        self,
        intrinsics: Optional[Intrinsics] = None,
        marker_size_m: Optional[float] = None,      # physical side length (meters)
        aruco_dict_id: int = cv2.aruco.DICT_6X6_250,
        res: Tuple[int, int] = (960, 720),          # match your working script defaults
        fps: int = 30,
    ):
        self.intrinsics = intrinsics
        self.marker_size_m = marker_size_m
        self.res = res
        self.fps = fps

        self._dict_id = aruco_dict_id
        self._dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self._params = cv2.aruco.DetectorParameters_create()  # default params (as in your working code)

        self._picam2: Optional[Picamera2] = None
        self._started = False

    # ----- Camera (Picamera2 only) -----

    def start_camera(self) -> None:
        if self._started:
            return
        w, h = self.res
        frame_dur = int(1.0 / self.fps * 1_000_000)
        cam = Picamera2()
        cfg = cam.create_video_configuration(
            main={"size": (w, h), "format": "RGB888"},
            controls={"FrameDurationLimits": (frame_dur, frame_dur)},
            queue=False,
        )
        cam.configure(cfg)
        cam.start()
        time.sleep(0.8)  # match your working script warm-up
        self._picam2 = cam
        self._started = True

    def read(self):
        """Return (ok, frame_bgr)."""
        if not self._started:
            self.start_camera()
        rgb = self._picam2.capture_array("main")
        return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def stop_camera(self):
        if self._picam2:
            try: self._picam2.stop()
            except: pass
            self._picam2 = None
        self._started = False

    # ----- Detection / pose -----

    @staticmethod
    def _center_xy(c4x2: np.ndarray) -> Tuple[float, float]:
        c = np.mean(c4x2.reshape(4, 2), axis=0)
        return float(c[0]), float(c[1])

    @staticmethod
    def _vertical_mean_px(pts: np.ndarray) -> float:
        # pts ordered TL, TR, BR, BL
        TL, TR, BR, BL = pts
        v1 = np.linalg.norm(TL - BL)
        v2 = np.linalg.norm(TR - BR)
        return float(0.5 * (v1 + v2))

    @staticmethod
    def _perimeter_px(pts: np.ndarray) -> float:
        TL, TR, BR, BL = pts
        return float(
            np.linalg.norm(TL-TR) + np.linalg.norm(TR-BR) +
            np.linalg.norm(BR-BL) + np.linalg.norm(BL-TL)
        )

    def detect(
        self,
        frame_bgr,
        restrict_ids: Optional[List[int]] = None,
        want_pose: Optional[bool] = None,
    ) -> List[Detection]:
        """
        Detect markers using DICT_6X6_250 directly on BGR (no grayscale step),
        choose the largest by perimeter (as in your working code), and (optionally)
        estimate rvec/tvec if intrinsics+marker_size_m are set.
        """
        if want_pose is None:
            want_pose = (self.intrinsics is not None and self.marker_size_m is not None)

        corners_list, ids, _ = cv2.aruco.detectMarkers(
            frame_bgr, self._dict, parameters=self._params
        )
        if ids is None or len(corners_list) == 0:
            return []

        # pick the largest by perimeter, but return all (your state machine may choose largest itself)
        # we’ll compute pose vectorized if requested
        rvecs = tvecs = None
        if want_pose:
            K = np.array([[self.intrinsics.fx, 0, self.intrinsics.cx],
                          [0, self.intrinsics.fy, self.intrinsics.cy],
                          [0, 0, 1]], dtype=np.float32)
            dist = self.intrinsics.dist.reshape(-1, 1).astype(np.float32)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners_list, float(self.marker_size_m), K, dist
            )

        dets: List[Detection] = []
        for i, c in enumerate(corners_list):
            mid = int(ids[i][0])
            if restrict_ids is not None and mid not in restrict_ids:
                continue
            pts = c.reshape(-1, 2).astype(np.float32)  # TL, TR, BR, BL
            dets.append(
                Detection(
                    marker_id=mid,
                    corners=pts,
                    center_xy=self._center_xy(pts),
                    side_px=self._vertical_mean_px(pts),
                    rvec=(rvecs[i].reshape(3) if rvecs is not None else None),
                    tvec=(tvecs[i].reshape(3) if tvecs is not None else None),
                )
            )
        return dets

    @staticmethod
    def choose_largest(dets: List[Detection]) -> Optional[Detection]:
        """Select detection with the largest *perimeter*, matching your working code’s heuristic."""
        if not dets:
            return None
        return max(dets, key=lambda d: ArucoUtils._perimeter_px(d.corners))

    @staticmethod
    def yaw_from_tvec(tvec: np.ndarray) -> float:
        """Yaw error (rad): + if marker is to the right of camera forward."""
        x, _, z = float(tvec[0]), float(tvec[1]), float(tvec[2])
        return float(np.arctan2(x, z))

    # ----- Robot call-through steps (use your robot API elsewhere) -----

    @staticmethod
    def rotate_step(bot, angle_deg: float, speed: Optional[int] = None) -> None:
        bot.turn_angle(angle_deg, speed=speed)

    @staticmethod
    def forward_step(bot, meters: float, speed: Optional[int] = None) -> None:
        bot.drive_distance(meters, direction=bot.FORWARD, speed=speed)
