# LocalizationPathing.py  (drop-in)
import time
import math
import numpy as np

class LocalizationPathing:
    """
    Minimal + fast:
      - Explore until each required landmark ID has been seen at least once.
      - Then: rotate toward midpoint and drive in chunky steps.

    TURN SIGN: We assume robot.turn_angle(+deg) turns LEFT/CCW.
               So we send the SAME SIGN as the math heading error.
    """

    def __init__(self, robot, camera, required_landmarks, step_cm=30, rotation_deg=15):
        self.robot = robot
        self.camera = camera
        self.required_landmarks = set(required_landmarks)
        self.step_cm = float(step_cm)
        self.rotation_deg = float(rotation_deg)

        self.observed_landmarks = set()
        self.all_seen = False

        # Tiny alignment deadband so we don't waste time micro-turning
        self.align_deadband_rad = math.radians(3.0)

    # --------------------------- Exploration ---------------------------
    def explore_step(self, drive=False, min_dist=400):
        """
        Quick scan/step to see tags. Returns (distance_cm, angle_rad_math).
        We keep this short so it doesn't 'think' forever.
        """
        dist = 0.0
        angle_deg = self.rotation_deg

        if self.all_seen:
            return 0.0, 0.0

        if not drive:
            # Quick small left/CCW nudge
            self.robot.turn_angle(angle_deg)
            angle_rad = math.radians(angle_deg)   # SAME SIGN as command
            # no long sleeps; camera read happens below
        else:
            dist = self.step_cm
            left, center, right = self.robot.proximity_check()
            if left < min_dist or center < min_dist or right < min_dist:
                self.robot.stop()

            # Quick bias away from closer side
            if left > right:
                self.robot.turn_angle(+20)         # small, quick
                angle_rad = math.radians(+20)
            else:
                self.robot.turn_angle(-20)
                angle_rad = math.radians(-20)

            self.robot.drive_distance_cm(dist)

        # Check for tags after the motion
        frame = self.camera.get_next_frame()
        objectIDs, dists, angles = self.camera.detect_aruco_objects(frame)
        if objectIDs is not None:
            self.observed_landmarks.update(objectIDs)
        self.all_seen = self.required_landmarks.issubset(self.observed_landmarks)

        return dist, angle_rad if 'angle_rad' in locals() else 0.0

    def seen_all_landmarks(self):
        return self.all_seen

    # ---------------------- Point-and-go to midpoint ----------------------
    def move_towards_goal_step(self, est_pose, center, step_cm=None):
        """
        Rotate toward midpoint (same-sign command as math error), then drive.
        Returns (distance_cm, angle_rad_applied).
        """
        if step_cm is None:
            step_cm = self.step_cm

        # Pose and goal
        rx, ry = float(est_pose.getX()), float(est_pose.getY())
        rth    = float(est_pose.getTheta())
        gx, gy = float(center[0]), float(center[1])

        dx, dy = gx - rx, gy - ry
        dist_to_goal = float(math.hypot(dx, dy))
        if dist_to_goal < 5.0:
            print("reached center")
            return 0.0, 0.0

        # Math heading error (CCW +), wrap to [-pi, pi]
        angle_to_center = math.atan2(dy, dx) - rth
        angle_to_center = math.atan2(math.sin(angle_to_center), math.cos(angle_to_center))

        # If notably misaligned, ROTATE FIRST, no negation:
        if abs(angle_to_center) > self.align_deadband_rad:
            self.robot.turn_angle(math.degrees(angle_to_center))  # SAME SIGN
            return 0.0, angle_to_center

        # Then DRIVE in a decent chunk toward the goal
        move_distance = float(min(step_cm, dist_to_goal))
        self.robot.drive_distance_cm(move_distance)
        return move_distance, 0.0
