import matplotlib.pyplot as plt
from RobotUtils.CameraUtils import CameraUtils, ArucoUtils
from RobotUtils.LandmarkUtils import LandmarkUtils
from LandmarkOccupancyGrid import LandmarkOccupancyGrid
from robot_model import RobotModel
from robot_RRT import robot_RRT
from RobotUtils.CalibratedRobot import CalibratedRobot

cam = CameraUtils()
cam.start_camera()
aruco = ArucoUtils()
landmark_utils = LandmarkUtils(cam, aruco)
grid_map = LandmarkOccupancyGrid(low=(-2,-1), high=(1.5, 5), res=0.05)
cal_arlo = CalibratedRobot()

landmarks = landmark_utils.map_landmarks()
print(f"landmarks detected: {landmarks}")


landmark_radius = 0.20
landmarks_for_grid = [(pos[0], pos[1], landmark_radius) for _, pos in landmarks]
                      
grid_map.add_landmarks(landmarks_for_grid)

robot = RobotModel()

path_res = 0.05

rrt = robot_RRT(
    start=[0, 0],
    goal=[0, 4],
    robot_model=robot,
    map=grid_map,
    expand_dis=0.2,
    path_resolution=path_res,
    )

path =rrt.planning()


if path is None:
    print("Cannot find path")
else:
    print(f"path: {path}")
    cal_arlo.follow_path(path)
    rrt.draw_graph(path, name = "1")
    print("found path")