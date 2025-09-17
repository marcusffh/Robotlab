#!/usr/bin/env python3
# Picamera2-only ArUco utilities (BGR, DICT_6X6_250; no grayscale, no fallbacks).
# Mirrors your working detector and exposes simple helpers.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import time
import numpy as np
import cv2
from picamera2 import Picamera2  # hard requirement


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray  # kept for future pose mode (not used here)


class ArucoUtils:
    def __init__(self, res: Tuple[int, int] = (960, 720), fps: int = 30):
        # 960x720 main stream => sensor still runs 1640x1232 internally; ISP downsamples nicely.
        self.res = res
        self.fps = fps
        self._picam2: Optional[Picamera2] = None
        self._started = False

        # Exact dictionary + default params (as in your working code)
        self._dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self._params = cv2.aruco.DetectorParameters_create()

    # ---------- Camera (Picamera2 only) ----------
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
        time.sleep(0.8)  # same warm-up that worked for you
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

    # ---------- Detection (your exact logic) ----------
    def detect_one(self, frame_bgr, restrict_id: Optional[int] = None):
        """
        Detect DICT_6X6_250 on BGR and return:
            { 'id': int, 'x_px': float, 'cx': float, 'w': int }
        or None if not found. Chooses the largest by perimeter (stable).
        """
        aruco = cv2.aruco
        corners_list, ids, _ = aruco.detectMarkers(frame_bgr, self._dict, parameters=self._params)
        if ids is None or len(corners_list) == 0:
            return None

        best = None
        for c, mid in zip(corners_list, ids.flatten()):
            mid = int(mid)
            if restrict_id is not None and mid != restrict_id:
                continue
            pts = c.reshape(-1, 2)  # TL, TR, BR, BL
            per = (
                np.linalg.norm(pts[0] - pts[1]) +
                np.linalg.norm(pts[1] - pts[2]) +
                np.linalg.norm(pts[2] - pts[3]) +
                np.linalg.norm(pts[3] - pts[0])
            )
            if best is None or per > best[0]:
                best = (per, mid, pts)

        if best is None:
            return None

        _, mid, pts = best
        TL, TR, BR, BL = pts
        v1 = np.linalg.norm(TL - BL)
        v2 = np.linalg.norm(TR - BR)
        x_px = 0.5 * (v1 + v2)               # vertical pixel height (mean of both sides)
        cx   = float((TL[0]+TR[0]+BR[0]+BL[0]) / 4.0)
        h, w = frame_bgr.shape[:2]
        return {"id": mid, "x_px": float(x_px), "cx": cx, "w": w}

    # ---------- Tiny helpers (call your CalibratedRobot API) ----------
    @staticmethod
    def rotate_step(bot, angle_deg: float, speed: Optional[int] = None) -> None:
        """
        Positive angle = left, negative = right (matches your CalibratedRobot.turn_angle).
        """
        bot.turn_angle(angle_deg, speed=speed)

    @staticmethod
    def forward_step(bot, meters: float, speed: Optional[int] = None) -> None:
        bot.drive_distance(meters, direction=bot.FORWARD, speed=speed)

    # Back-compat alias if another script used this name
    @staticmethod
    def go_forward_step(bot, meters: float, speed: Optional[int] = None) -> None:
        ArucoUtils.forward_step(bot, meters, speed)
