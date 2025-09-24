# map_utils.py
# Class-based utilities for local mapping from ArUco detections (2D x–z plane)

import time
import math
from collections import defaultdict
import numpy as np
import cv2


# Robotutils/mapping_utils.py
import time
import math
from collections import defaultdict
import numpy as np
import cv2


class LocalMapper:
    def __init__(self,
                 extent_m: float = 5.0,
                 grid_res_m: float = 0.05,
                 landmark_radius_m: float = 0.17,
                 robot_radius_m: float = 0.23):
        self.extent_m = float(extent_m)
        self.grid_res_m = float(grid_res_m)
        self.landmark_radius_m = float(landmark_radius_m)
        self.robot_radius_m = float(robot_radius_m)


    def visualize_grid(self, landmarks, scale=2, save_path="local_map.png", path=None):
        """
        Save local map visualization to a PNG file (no GUI shown).
        White=free, black=landmark, gray=inflation, red circle=origin.
        If path is given, overlay as a red polyline with green (start) + blue (goal).
        """
        grid, inflated, origin = self.build_grid_from_landmarks(landmarks)
        vis = np.full(grid.shape, 255, np.uint8)
        vis[grid > 0] = 0
        vis[(inflated > 0) & (grid == 0)] = 128
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # draw robot at origin
        cv2.circle(vis, (origin[1], origin[0]),
                max(1, int(self.robot_radius_m / self.grid_res_m)),
                (0, 0, 255), 1)

    # draw landmark centers
        for (_, lx, lz, _) in landmarks:
            idx = self._metric_to_idx(lx, lz, origin)
            if idx is not None:
                cv2.circle(vis, (idx[1], idx[0]), 2, (0, 255, 0), -1)

    # overlay path if provided
        if path and len(path) > 1:
            pts = []
            for (x, z) in path:
                idx = self._metric_to_idx(x, z, origin)
                if idx is not None:
                    pts.append((idx[1], idx[0]))  # (col,row)
            if len(pts) >= 2:
                cv2.polylines(vis, [np.array(pts, dtype=np.int32)], False, (0, 0, 255), 1)
                cv2.circle(vis, pts[0], 3, (0, 255, 0), -1)   # start
                cv2.circle(vis, pts[-1], 3, (255, 0, 0), -1)  # goal

        if scale != 1:
            vis = cv2.resize(vis, (vis.shape[1]*scale, vis.shape[0]*scale),
                            interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(save_path, vis)
        return vis




    # Landmark accumulation (camera)

    def accumulate_landmarks(self, camera_utils, aruco_utils, n_frames=10, sleep_s=0.03):
        """
        Accumulate detections over a few frames for robustness.
        Returns: list[(id, x_m, z_m, dist_m)], sorted by distance.
        - camera_utils: instance of your CameraUtils (already started)
        - aruco_utils: instance of your ArucoUtils (with marker_length set)
        """
        buf = defaultdict(list)
        K = camera_utils.camera_matrix
        dist = camera_utils.dist

        for _ in range(n_frames):
            frame_bgr = camera_utils.get_frame()
            corners, ids = aruco_utils.detect_markers(frame_bgr)
            if ids is not None and len(ids) > 0:
                rvecs, tvecs = aruco_utils.estimate_pose(corners, K, dist_coeffs=dist)
                for i in range(len(ids)):
                    t = tvecs[i, 0, :]  # (tx, ty, tz) in meters (camera frame)
                    x_m, z_m = float(t[0]), float(t[2])
                    d = float(np.linalg.norm(t))
                    buf[int(ids[i][0])].append((x_m, z_m, d))
            time.sleep(sleep_s)

        # reduce per-id by median
        landmarks = []
        for lid, arr in buf.items():
            xs = np.array([a[0] for a in arr], dtype=np.float32)
            zs = np.array([a[1] for a in arr], dtype=np.float32)
            ds = np.array([a[2] for a in arr], dtype=np.float32)
            landmarks.append((lid, float(np.median(xs)), float(np.median(zs)), float(np.median(ds))))

        landmarks.sort(key=lambda r: r[3])
        return landmarks


    # Circle-based landmark map (analytic 2D)

    def collision_circle_map(self, point_xz, landmarks):
        """
        Fast analytic collision: each landmark is a circle (landmark_radius_m),
        robot is a circle (robot_radius_m). Returns True if collision.
        """
        px, pz = point_xz
        R = self.landmark_radius_m + self.robot_radius_m
        R2 = R * R
        for (_, lx, lz, _) in landmarks:
            dx = px - lx
            dz = pz - lz
            if dx*dx + dz*dz <= R2:
                return True
        return False


    # Occupancy grid build

    def build_grid_from_landmarks(self, landmarks):
        """
        Build occupancy grid and an inflated version (dilated by robot radius).
        Returns: grid (uint8), grid_inflated (uint8), origin_idx (row,col)
        """
        grid, origin = self._make_grid()
        for (_, lx, lz, _) in landmarks:
            self._stamp_circle_on_grid(grid, origin, lx, lz, self.landmark_radius_m)

        # Inflate by robot radius using morphological dilation
        kernel_diam_cells = max(1, int(round(2 * self.robot_radius_m / self.grid_res_m)))
        if kernel_diam_cells % 2 == 0:
            kernel_diam_cells += 1  # make odd
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_diam_cells, kernel_diam_cells))
        inflated = cv2.dilate(grid, kernel)
        return grid, inflated, origin

    def collision_grid_map(self, point_xz, grid_inflated, origin_idx):
        """
        Collision check against inflated occupancy grid.
        Outside map => treated as collision (True).
        """
        idx = self._metric_to_idx(point_xz[0], point_xz[1], origin_idx)
        if idx is None:
            return True
        r, c = idx
        return grid_inflated[r, c] > 0



    # Internal helpers

    def _make_grid(self):
        side = int(np.ceil(2 * self.extent_m / self.grid_res_m))
        grid = np.zeros((side, side), dtype=np.uint8)  # 0=free, 1=occupied
        origin_idx = (side // 2, side // 2)
        return grid, origin_idx

    def _metric_to_idx(self, x_m, z_m, origin_idx):
        row0, col0 = origin_idx
        r = int(round(row0 - (x_m / self.grid_res_m)))
        c = int(round(col0 + (z_m / self.grid_res_m)))
        H = row0 * 2
        W = col0 * 2
        if r < 0 or r >= H or c < 0 or c >= W:
            return None
        return (r, c)

    def _stamp_circle_on_grid(self, grid, origin_idx, x_m, z_m, radius_m):
        idx = self._metric_to_idx(x_m, z_m, origin_idx)
        if idx is None:
            return
        rr, cc = idx
        r_cells = int(math.ceil(radius_m / self.grid_res_m))
        H, W = grid.shape
        r0 = max(0, rr - r_cells)
        r1 = min(H - 1, rr + r_cells)
        c0 = max(0, cc - r_cells)
        c1 = min(W - 1, cc + r_cells)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                x_cell = (origin_idx[0] - r) * self.grid_res_m
                z_cell = (c - origin_idx[1]) * self.grid_res_m
                if (x_cell - x_m)**2 + (z_cell - z_m)**2 <= radius_m**2:
                    grid[r, c] = 1
