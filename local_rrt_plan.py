from Robotutils.CameraDetection_util import CameraUtils, ArucoUtils
from Robotutils.CalibratedRobot import CalibratedRobot
from Robotutils.mapping_utils import LocalMapper
from Robotutils.rrt_utils import RRTPlanner

RES_W, RES_H, FPS = 1640, 1232, 30
MARKER_LEN_M = 0.14

def main():
    bot = CalibratedRobot()
    cam = CameraUtils(width=RES_W, height=RES_H, fx=1360, fy=1360)
    aruco = ArucoUtils(marker_length=MARKER_LEN_M)
    cam.start_camera()

    mapper = LocalMapper()
    try:
        landmarks = mapper.accumulate_landmarks(cam, aruco)
        grid, inflated, origin = mapper.build_grid_from_landmarks(landmarks)

        start = (0.0, 0.0)      # robot at origin
        goal = (0.0, 3.0)       # distance ahead (3m)

        rrt = RRTPlanner()
        path = rrt.plan(start, goal, mapper, inflated, origin)

        if path:
            print("RRT path:", path)
            mapper.visualize_grid(landmarks, scale=2, save_path="rrt_map.png", path=path)
            print("Saved rrt_map.png with path overlay")
        else:
            print("No path found.")
            mapper.visualize_grid(landmarks, scale=2, save_path="rrt_map.png")


        # Optional: drive robot along path
        # for each segment: turn_angle, drive_distance

    finally:
        cam.stop_camera()
        bot.stop()

if __name__ == "__main__":
    main()
