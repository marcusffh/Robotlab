# Robotutils/localmap_utils.py
# Ex4 Part 1: build a local map (inflated landmark circles) + collision checks.
import math

# Try Robotutils-first, then repo root fallbacks. Expect a get_landmarks() -> [(id,x,y,r?)] in meters.
def _import_get_landmarks():
    for mod_name in (
        "Robotutils.simple_landmark", "Robotutils.aruco_utils", "Robotutils.detect",
        "aruco_utils", "simple_landmark",
    ):
        try:
            mod = __import__(mod_name, fromlist=["get_landmarks"])
            fn = getattr(mod, "get_landmarks", None)
            if callable(fn):
                return fn
        except Exception:
            pass
    raise ImportError("No get_landmarks() found. Provide one in Robotutils/ or repo root.")

_get_lm = _import_get_landmarks()

def _norm(raw):
    out = []
    for e in raw:
        if isinstance(e, dict):
            out.append((int(e["id"]), float(e["x"]), float(e["y"]), float(e.get("r", 0.12))))
        else:
            if len(e) == 4: _id, x, y, r = e; out.append((int(_id), float(x), float(y), float(r)))
            elif len(e) == 3: _id, x, y = e; out.append((int(_id), float(x), float(y), 0.12))
    return out

def get_circles(robot_radius=0.18):
    """Returns inflated circles as [(cx, cy, R)], R = landmark_r + robot_radius."""
    return [(x, y, r + float(robot_radius)) for (_id, x, y, r) in _norm(_get_lm())]

def point_in_collision(x, y, circles):
    """True iff (x,y) hits any inflated circle."""
    return any(math.hypot(x - cx, y - cy) <= R for (cx, cy, R) in circles)

def segment_in_collision(p, q, circles, step=0.02):
    """True iff straight line p->q hits any inflated circle (sampled)."""
    (x0, y0), (x1, y1) = p, q
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy)
    if d == 0: return point_in_collision(x0, y0, circles)
    n = max(1, int(d / step))
    for i in range(n + 1):
        t = i / n
        if point_in_collision(x0 + t*dx, y0 + t*dy, circles):
            return True
    return False
# --- Minimal occupancy grid utils (Part 1 option A) --------------------------
class Grid:
    """Boolean occupancy grid in a local window around the robot."""
    def __init__(self, width_m=6.0, height_m=6.0, resolution_m=0.05, origin_xy_m=(-3.0, -3.0)):
        self.res = float(resolution_m)
        self.w = int(round(width_m / self.res))
        self.h = int(round(height_m / self.res))
        self.ox, self.oy = origin_xy_m
        self._g = [[False for _ in range(self.w)] for _ in range(self.h)]

    def world_to_grid(self, x, y):
        gx = int((x - self.ox) // self.res)
        gy = int((y - self.oy) // self.res)
        if 0 <= gx < self.w and 0 <= gy < self.h:
            return gx, gy
        return None

    def _stamp_disc(self, cx, cy, R):
        if R <= 0: return
        r2 = R * R
        # Bound box (clamped to grid)
        gmin = self.world_to_grid(cx - R, cy - R) or (0, 0)
        gmax = self.world_to_grid(cx + R, cy + R) or (self.w - 1, self.h - 1)
        gx0, gy0 = max(0, gmin[0]), max(0, gmin[1])
        gx1, gy1 = min(self.w - 1, gmax[0]), min(self.h - 1, gmax[1])
        for gy in range(gy0, gy1 + 1):
            yc = self.oy + (gy + 0.5) * self.res
            for gx in range(gx0, gx1 + 1):
                xc = self.ox + (gx + 0.5) * self.res
                if (xc - cx) ** 2 + (yc - cy) ** 2 <= r2:
                    self._g[gy][gx] = True

    def occupied(self, x, y):
        g = self.world_to_grid(x, y)
        if g is None:
            return True  # out of map = blocked
        gx, gy = g
        return self._g[gy][gx]

    def to_numpy(self):
        try:
            import numpy as np
            return np.array(self._g, dtype=bool)
        except Exception:
            return None

def build_grid(circles, width_m=6.0, height_m=6.0, resolution_m=0.05, origin_xy_m=(-3.0, -3.0)):
    """
    circles: [(cx, cy, R)] in meters (already inflated).
    Returns: Grid() with discs stamped in.
    """
    g = Grid(width_m, height_m, resolution_m, origin_xy_m)
    for (cx, cy, R) in circles:
        g._stamp_disc(cx, cy, R)
    return g
