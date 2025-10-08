# explore_landmarks.py
from RobotUtils.CalibratedRobot import CalibratedRobot
import camera
import time
import numpy as np
import math

class LocalizationPathing:
    def __init__(self, robot, camera, required_landmarks, step_cm=20, rotation_deg=20):
        self.robot = robot
        self.camera = camera
        self.required_landmarks = set(required_landmarks)
        self.step_cm = step_cm
        self.rotation_deg = rotation_deg

        self.observed_landmarks = set()
        self.all_seen = False

        # small, safe alignment deadband (don’t drive until roughly facing goal)
        self.align_deadband_rad = math.radians(5.0)

    def explore_step(self, drive=False, min_dist=400):
        dist = 0.0
        angle_deg = float(self.rotation_deg)

        # IMPORTANT: robot +deg = CW (right) -> math angle is NEGATIVE radians
        angle_rad = -math.radians(angle_deg)

        if self.all_seen:
            return 0.0, 0.0

        if not drive:
            self.robot.turn_angle(angle_deg)       # CW
            time.sleep(0.2)
        else:
            dist = float(self.step_cm)
            left, center, right = self.robot.proximity_check()

            if left < min_dist or center < min_dist or right < min_dist:
                self.robot.stop()

            if left > right:
                self.robot.turn_angle(45)          # CW
                angle_rad = -math.radians(45)      # math negative
            else:
                self.robot.turn_angle(-45)         # CCW
                angle_rad = +math.radians(45)      # math positive

            self.robot.drive_distance_cm(dist)

        # Update observed landmarks
        frame = self.camera.get_next_frame()
        objectIDs, dists, angles = self.camera.detect_aruco_objects(frame)
        if objectIDs is not None:
            self.observed_landmarks.update(objectIDs)

        self.all_seen = self.required_landmarks.issubset(self.observed_landmarks)
        return dist, angle_rad

    def seen_all_landmarks(self):
        """Returns True if all required landmarks have been observed."""
        return self.all_seen

    def move_towards_goal_step(self, est_pose, center, step_cm=10000):
        robot_pos = np.array([est_pose.getX(), est_pose.getY()], dtype=float)
        direction = np.array(center, dtype=float) - robot_pos
        distance_to_center = float(np.linalg.norm(direction))

        # compute math heading error (CCW positive), wrap to [-pi, pi]
        angle_to_center = math.atan2(direction[1], direction[0]) - float(est_pose.getTheta())
        angle_to_center = math.atan2(math.sin(angle_to_center), math.cos(angle_to_center))

        # close enough?
        if distance_to_center < 5.0:
            print("reached center")
            return 0.0, 0.0

        # FIRST: if notably misaligned, rotate only (don’t drive this tick)
        if abs(angle_to_center) > self.align_deadband_rad:
            # KEY FIX: robot +deg = CW, so send the NEGATIVE of the math angle
            self.robot.turn_angle(-math.degrees(angle_to_center))
            # We applied 'angle_to_center' in math space
            return 0.0, angle_to_center

        # THEN: drive forward (bounded by remaining distance)
        move_distance = float(min(step_cm, distance_to_center))

        print(f"distance moved: {move_distance:.1f} cm")
        print(f"heading error (rad): {angle_to_center:.3f}")

        self.robot.drive_distance_cm(move_distance)

        # We drove straight; rotation applied this tick is ~0
        return move_distance, 0.0
