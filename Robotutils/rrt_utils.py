# Robotutils/rrt_utils.py
import math, random

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
