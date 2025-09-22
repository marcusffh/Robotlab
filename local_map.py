# local_map.py
import cv2
from Robotutils.CalibratedRobot import CalibratedRobot
from Robotutils.CameraDetection_util import CameraUtils, ArucoUtils
from Robotutils.mapping_utils import LocalMapper

RES_W, RES_H, FPS = 1640, 1232, 30
MARKER_LEN_M = 0.14

def main():
    bot = CalibratedRobot()
    cam = CameraUtils(width=RES_W, height=RES_H, fx=1360, fy=1360)
    aruco = ArucoUtils(marker_length=MARKER_LEN_M)
    cam.start_camera(RES_W, RES_H, FPS)

    mapper = LocalMapper(
        extent_m=3.0,
        grid_res_m=0.05,
        landmark_radius_m=0.09,
        robot_radius_m=0.12
    )

    try:
        landmarks = mapper.accumulate_landmarks(cam, aruco, n_frames=10, sleep_s=0.03)
        print("\nDetected landmarks (id, x[m], z[m], dist[m]):")
        for rec in landmarks:
            print(" ", rec)

        # Save visualization instead of showing GUI
        mapper.visualize_grid(landmarks, scale=2, save_path="local_map.png")
        print("Local map saved to local_map.png")

        # Example collision checks
        test_pts = [(0.0, 0.5), (0.2, 0.4), (0.5, 1.0)]
        print("\nCollision checks:")
        grid, inflated, origin = mapper.build_grid_from_landmarks(landmarks)
        for p in test_pts:
            c_circle = mapper.collision_circle_map(p, landmarks)
            c_grid = mapper.collision_grid_map(p, inflated, origin)
            print(f"  {p} -> circle: {'HIT' if c_circle else 'free'} | grid: {'HIT' if c_grid else 'free'}")

    finally:
        cam.stop_camera()
        bot.stop()

if __name__ == "__main__":
    main()
