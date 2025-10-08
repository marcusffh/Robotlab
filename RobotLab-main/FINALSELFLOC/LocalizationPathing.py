# explore_landmarks.py
from RobotUtils.CalibratedRobot import CalibratedRobot
import camera
import time
import numpy as np

import time

class LocalizationPathing:
    def __init__(self, robot, camera, required_landmarks, step_cm=20, rotation_deg=20):
        self.robot = robot
        self.camera = camera
        self.required_landmarks = set(required_landmarks)
        self.step_cm = step_cm
        self.rotation_deg = rotation_deg

        self.observed_landmarks = set()
        self.all_seen = False

    def explore_step(self, drive=False, min_dist = 400):
        dist = 0
        angle_deg = self.rotation_deg 
        angle_rad = np.radians(angle_deg)

        if self.all_seen:
            return 0, 0 

        if not drive:
            self.robot.turn_angle(angle_deg)
            time.sleep(0.2)

        if drive:
            dist = self.step_cm
            left, center, right = self.robot.proximity_check()

            if left < min_dist or center < min_dist or right < min_dist:
                self.robot.stop()
            if left > right:
                self.robot.turn_angle(45)   
                angle_rad = np.radians(45)
            else:
                self.robot.turn_angle(-45)
                angle_rad = np.radians(-45)

            self.robot.drive_distance_cm(dist)

        frame = self.camera.get_next_frame()
        objectIDs, dists, angles = self.camera.detect_aruco_objects(frame)
        if objectIDs is not None:
            self.observed_landmarks.update(objectIDs)

        self.all_seen = self.required_landmarks.issubset(self.observed_landmarks)

        return dist, angle_rad


    def seen_all_landmarks(self):
        """
        Returns True if all required landmarks have been observed.
        """
        return self.all_seen
    
    def move_towards_goal_step(self, est_pose, center, step_cm=10000):
        robot_pos = np.array([est_pose.getX(), est_pose.getY()])
        direction = center - robot_pos
        distance_to_center = np.linalg.norm(direction)
        angle_to_center = np.arctan2(direction[1], direction[0]) - est_pose.getTheta()

        if distance_to_center < 5:
            print("reached center")
            return 0, 0
        
        angle_to_center = (angle_to_center + np.pi) % (2 * np.pi) - np.pi
        
        move_distance = min(step_cm, distance_to_center)

        print(f"distance moved: {move_distance}")
        print(f"angle (rad) turned: {angle_to_center}")

        self.robot.turn_angle(np.degrees(angle_to_center))

        self.robot.drive_distance_cm(move_distance)

        return move_distance, angle_to_center


