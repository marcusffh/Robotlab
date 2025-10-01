# local_rrt_plan.py
from Robotutils.CameraDetection_util import CameraUtils, ArucoUtils
from Robotutils.CalibratedRobot import CalibratedRobot
from Robotutils.mapping_utils import LocalMapper
from Robotutils.rrt_utils import RRTPlanner, smooth_path
from Robotutils.path_execute_utils import execute_path

def main():
    bot = CalibratedRobot()
    cam = CameraUtils()
    aruco = ArucoUtils()
    cam.start_camera()

    mapper = LocalMapper(robot_radius_m=0.12 + 0.10 + 0.05)
    try:
        # build local occupancy map 
        landmarks = mapper.accumulate_landmarks(cam, aruco)
        grid, inflated, origin = mapper.build_grid_from_landmarks(landmarks)

        # Plan with local RRT 
        start = (0.0, 0.0)      # robot at origin
        goal  = (0.0, 4.5)      # keep within extent
        rrt = RRTPlanner(step_size=0.20, max_iters=1500, goal_sample_rate=0.15, goal_radius=0.25)
        path = rrt.plan(start, goal, mapper, inflated, origin)

        if path:
            print("RRT raw path:", path)
            mapper.visualize_grid(landmarks, scale=2, save_path="rrt_map_raw.png", path=path)
            print("Saved rrt_map_raw.png")

            # Smooth the path (shortcutting + pruning)
            smoothed = smooth_path(
                path, mapper, inflated, origin,
                step_m=0.03,       # collision sampling along shortcuts
                min_turn_deg=15,   # raise for fewer waypoints
                min_seg_len=0.12,  # skip tiny segments
                rounds=3
            )
            print("Smoothed path:", smoothed)
            mapper.visualize_grid(landmarks, scale=2, save_path="rrt_map_smoothed.png", path=smoothed)
            print("Saved rrt_map_smoothed.png")

            # Execute the path on the robot
            path_to_run = smoothed if smoothed else path
            print("Executing path...")
            
            execute_path(
                bot,
                path_to_run,
                drive_speed=64,
                turn_speed=64,
                heading_tol_deg=3.0,
                min_seg_len=0.05,
                settle_s=0.05
            )
            print("Path execution complete.")
        else:
            print("No path found.")
            mapper.visualize_grid(landmarks, scale=2, save_path="rrt_map_raw.png")
            print("Saved rrt_map_raw.png (no path).")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cam.stop_camera()
        bot.stop()

if __name__ == "__main__":
    main()
