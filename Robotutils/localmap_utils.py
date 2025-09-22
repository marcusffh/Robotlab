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
