# local_rrt_plan.py
from Robotutils.CameraDetection_util import CameraUtils, ArucoUtils
from Robotutils.CalibratedRobot import CalibratedRobot
from Robotutils.mapping_utils import LocalMapper
from Robotutils.rrt_utils import RRTPlanner, smooth_path

RES_W, RES_H, FPS = 1640, 1232, 30
MARKER_LEN_M = 0.14

def main():
    bot = CalibratedRobot()
    cam = CameraUtils(width=RES_W, height=RES_H, fx=1360, fy=1360)
    aruco = ArucoUtils(marker_length=MARKER_LEN_M)
    cam.start_camera()   # no args -> uses CameraUtils defaults

    mapper = LocalMapper()  # uses default extent=3.0m, grid_res=0.05, etc.
    try:
        # 1) Map landmarks
        landmarks = mapper.accumulate_landmarks(cam, aruco)
        grid, inflated, origin = mapper.build_grid_from_landmarks(landmarks)

        # 2) Plan with simple RRT
        start = (0.0, 0.0)      # robot at origin
        goal  = (0.0, 2.5)      # keep within extent (2.5m < 3.0m)
        rrt = RRTPlanner(step_size=0.20, max_iters=1500, goal_sample_rate=0.15, goal_radius=0.25)
        path = rrt.plan(start, goal, mapper, inflated, origin)

        if path:
            print("RRT raw path:", path)
            # Save raw overlay
            mapper.visualize_grid(landmarks, scale=2, save_path="rrt_map_raw.png", path=path)
            print("Saved rrt_map_raw.png")

            # 3) Smooth the path (shortcutting + pruning)
            smoothed = smooth_path(path, mapper, inflated, origin,
                                   step_m=0.03,      # collision sampling along shortcuts
                                   min_turn_deg=10,  # increase for fewer waypoints
                                   min_seg_len=0.12, # skip tiny segments
                                   rounds=2)
            print("Smoothed path:", smoothed)
            mapper.visualize_grid(landmarks, scale=2, save_path="rrt_map_smoothed.png", path=smoothed)
            print("Saved rrt_map_smoothed.png")

            # 4) (optional) execute smoothed path:
            # execute_path(bot, smoothed)

        else:
            print("No path found.")
            mapper.visualize_grid(landmarks, scale=2, save_path="rrt_map_raw.png")

    finally:
        cam.stop_camera()
        bot.stop()

if __name__ == "__main__":
    main()
