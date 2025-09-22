# Minimal Part 1 runner using your Robotutils imports + the new utils.
import Robotutils.robot as robot
from Robotutils.localmap_utils import (
    get_circles, point_in_collision, segment_in_collision,
    build_grid,  # NEW
)

ROBOT_RADIUS = 0.18  # adjust if your Arlo footprint differs

if __name__ == "__main__":
    # Same style as your project: make the robot object
    arlo = robot.Robot()

    # Build the local map (inflated landmark circles) from your existing detector
    circles = get_circles(ROBOT_RADIUS)
    print("Inflated circles [(cx, cy, R)] in meters:\n", circles)

    # Tiny demo checks (you can delete these lines if you like)
    print("Collision at origin?", point_in_collision(0.0, 0.0, circles))
    print("Segment (0,0)->(1,0) collides?", segment_in_collision((0,0), (1,0), circles))

    # --- NEW: Build a small occupancy grid (Part 1 grid variant) ---
    grid = build_grid(circles, width_m=4.0, height_m=3.0, resolution_m=0.05, origin_xy_m=(-0.5, -1.5))
    print("Grid occupied at (1.0, 0.0)?", grid.occupied(1.0, 0.0))
