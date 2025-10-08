# LocalizationPathing.py
import time
import math
import numpy as np

class LocalizationPathing:
    """
    Minimal, deterministic pathing:
      - Explore (rotate/step) until each required landmark ID has been seen at least once.
      - Then: rotate fully to face the midpoint and drive straight toward it.
    
    Sign convention:
      - Robot API:  robot.turn_angle(+deg)  => RIGHT / CW
      - Math space: +radians                => CCW (left)
      => send -degrees(math_angle) to the robot to rotate CCW in math space.
    """

    def __init__(self, robot, camera, required_landmarks, step_cm=30.0, rotation_deg=25.0):
        self.robot = robot
        self.camera = camera
        self.required_landmarks = set(required_landmarks)

        self.step_cm = float(step_cm)          # forward chunk per call (cm)
        self.rotation_deg = float(rotation_deg)

        self.observed_landmarks = set()
        self.all_seen = False

    # ----------------------------- Exploration -----------------------------
    def explore_step(self, drive=False, min_dist=400):
        """
        Spin/step to find landmarks. Returns (distance_cm, angle_rad_math).
        angle_rad_math is CCW-positive (math convention).
        """
        dist_cm = 0.0
        angle_rad_math = 0.0

        if self.all_seen:
            return 0.0, 0.0

        if not drive:
            # Rotate in place to scan (robot +deg => CW; math angle is negative of that).
            self.robot.turn_angle(self.rotation_deg)
            angle_rad_math = -math.radians(self.rotation_deg)
            time.sleep(0.2)
        else:
            # Small forward step with a bias away from the closer side.
            dist_cm = float(self.step_cm)
            left, center, right = self.robot.proximity_check()

            if (left < min_dist) or (center < min_dist) or (right < min_dist):
                self.robot.stop()

            if left > right:
                self.robot.turn_angle(45)            # CW
                angle_rad_math = -math.radians(45)   # math negative
            else:
                self.robot.turn_angle(-45)           # CCW
                angle_rad_math = +math.radians(45)   # math positive

            self.robot.drive_distance_cm(dist_cm)

        # Update observed landmarks set
        frame = self.camera.get_next_frame()
        objectIDs, dists, angles = self.camera.detect_aruco_objects(frame)
        if objectIDs is not None:
            self.observed_landmarks.update(objectIDs)
        self.all_seen = self.required_landmarks.issubset(self.observed_landmarks)

        return dist_cm, angle_rad_math

    def seen_all_landmarks(self):
        return self.all_seen

    # --------------------- Point-and-go to midpoint step -------------------
    def move_towards_goal_step(self, est_pose, center, step_cm=None, center_tol_cm=10.0):
        """
        1) Rotate FULLY to face the midpoint.
        2) Drive straight toward it in chunks (step_cm).
        Stops driving if already within center_tol_cm.

        Returns: (distance_cm_commanded, angle_rad_math_applied)
        """
        if step_cm is None:
            step_cm = self.step_cm

        # Pose (cm, rad) and goal
        rx, ry = float(est_pose.getX()), float(est_pose.getY())
        rth    = float(est_pose.getTheta())
        gx, gy = float(center[0]), float(center[1])

        dx, dy = gx - rx, gy - ry
        dist_to_goal = float(math.hypot(dx, dy))

        # If within tolerance, don't move
        if dist_to_goal <= center_tol_cm:
            return 0.0, 0.0

        # Heading error in math convention (CCW positive)
        heading_error = math.atan2(dy, dx) - rth
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))  # wrap [-pi, pi]

        # --- FULL ROTATION to face the goal (one command) ---
        turn_deg_cmd = -math.degrees(heading_error)     # robot +deg = CW, so negate
        if abs(turn_deg_cmd) > 1.0:                     # ignore <1° jitter
            self.robot.turn_angle(turn_deg_cmd)
            applied_turn_math = heading_error
        else:
            applied_turn_math = 0.0

        # --- STRAIGHT DRIVE toward the goal in chunks ---
        move_distance = float(min(step_cm, dist_to_goal))
        if move_distance > 0.0:
            self.robot.drive_distance_cm(move_distance)

        # Return commands in math convention
        return move_distance, applied_turn_math
