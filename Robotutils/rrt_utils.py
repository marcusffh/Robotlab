# Robotutils/rrt_utils.py
import math, random
import numpy as np

class RRTPlanner:
    def __init__(self, step_size=0.2, max_iters=1000, goal_sample_rate=0.1, goal_radius=0.25):
        self.step = step_size
        self.max_iters = max_iters
        self.goal_rate = goal_sample_rate
        self.goal_radius = goal_radius

    def plan(self, start, goal, mapper, inflated, origin):
        """
        Plan path with RRT on mapper's occupancy grid.
        start, goal: (x,z) in meters
        mapper: LocalMapper instance
        inflated, origin: from build_grid_from_landmarks
        Returns: list of waypoints [(x,z), ...] or [] if fail
        """
        nodes = [start]
        parents = {0: -1}

        for _ in range(self.max_iters):
            # random sample (goal bias)
            if random.random() < self.goal_rate:
                q_rand = goal
            else:
                q_rand = ((random.random()*2 - 1)*mapper.extent_m,
                          (random.random()*2 - 1)*mapper.extent_m)

            # nearest
            i_near = min(range(len(nodes)), key=lambda i: self._dist(nodes[i], q_rand))

            # step toward
            q_new = self._steer(nodes[i_near], q_rand)

            # collision-free?
            if not mapper.collision_grid_map(q_new, inflated, origin):
                nodes.append(q_new)
                parents[len(nodes)-1] = i_near

                # goal reached?
                if self._dist(q_new, goal) <= self.goal_radius:
                    return self._reconstruct(nodes, parents, len(nodes)-1)

        return []

    def _steer(self, q_from, q_to):
        dx, dz = q_to[0]-q_from[0], q_to[1]-q_from[1]
        d = math.hypot(dx, dz)
        if d <= self.step:
            return q_to
        s = self.step / d
        return (q_from[0]+s*dx, q_from[1]+s*dz)

    def _dist(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def _reconstruct(self, nodes, parents, goal_i):
        path = []
        i = goal_i
        while i != -1:
            path.append(nodes[i])
            i = parents[i]
        return path[::-1]
    
# smooth path utils


def _edge_collision_grid(mapper, inflated, origin_idx, p, q, step_m=0.03):
    """Return True if segment p->q hits inflated obstacles (grid ray-march)."""
    dx, dz = q[0] - p[0], q[1] - p[1]
    dist = math.hypot(dx, dz)
    steps = max(2, int(math.ceil(dist / step_m)))
    for k in range(steps + 1):
        t = k / steps
        x = p[0] + t * dx
        z = p[1] + t * dz
        if mapper.collision_grid_map((x, z), inflated, origin_idx):
            return True
    return False

def shortcut_path(path, mapper, inflated, origin_idx, step_m=0.03):
    """Greedy line-of-sight shortcut: O(n^2) and very effective."""
    if not path or len(path) <= 2:
        return path[:]
    out = []
    i = 0
    n = len(path)
    while i < n:
        out.append(path[i])
        # try to jump as far as possible from i
        j = n - 1
        best = i + 1
        while j > i + 1:
            if not _edge_collision_grid(mapper, inflated, origin_idx, path[i], path[j], step_m):
                best = j
                break
            j -= 1
        i = best
        if i == n - 1:
            out.append(path[-1])
            break
    # remove possible duplicate next-to-last insertion
    if len(out) >= 2 and out[-1] == out[-2]:
        out.pop()
    return out

def _angle_deg(a, b, c):
    """Heading change at b given three points (x,z)."""
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    a1 = math.degrees(math.atan2(v1[0], v1[1]))
    a2 = math.degrees(math.atan2(v2[0], v2[1]))
    d = a2 - a1
    # wrap to [-180,180]
    while d > 180: d -= 360
    while d < -180: d += 360
    return abs(d)

def _seg_len(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])

def prune_angles(path, min_turn_deg=8.0, min_seg_len=0.12):
    """Remove tiny zigzags and micro segments."""
    if not path or len(path) <= 2:
        return path[:]
    out = [path[0]]
    for i in range(1, len(path) - 1):
        a, b, c = out[-1], path[i], path[i + 1]
        turn = _angle_deg(a, b, c)
        if turn < min_turn_deg and _seg_len(a, b) < min_seg_len:
            # skip b
            continue
        out.append(b)
    out.append(path[-1])
    return out

def smooth_path(path, mapper, inflated, origin_idx,
                step_m=0.03, min_turn_deg=8.0, min_seg_len=0.12, rounds=2):
    """Run shortcutting + angle pruning a couple of rounds."""
    if not path:
        return []
    p = path[:]
    for _ in range(rounds):
        p = shortcut_path(p, mapper, inflated, origin_idx, step_m=step_m)
        p = prune_angles(p, min_turn_deg=min_turn_deg, min_seg_len=min_seg_len)
    return p

