# Exercise4/simple_localmap.py  (no motion version)
from Robotutils.localmap_utils import (
    get_circles, point_in_collision, segment_in_collision, build_grid
)

ROBOT_RADIUS = 0.18

if __name__ == "__main__":
    circles = get_circles(ROBOT_RADIUS)  # purely vision-side; no driving
    print("Inflated circles:", circles)

    print("Collision at origin?", point_in_collision(0.0, 0.0, circles))
    print("Segment (0,0)->(1,0) collides?", segment_in_collision((0,0), (1,0), circles))

    grid = build_grid(circles, width_m=4.0, height_m=3.0, resolution_m=0.05, origin_xy_m=(-0.5,-1.5))
    print("Grid occupied at (1.0, 0.0)?", grid.occupied(1.0, 0.0))
