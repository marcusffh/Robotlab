# LocalizationPathing.py
import time
import math
import numpy as np

class LocalizationPathing:
    """
    Pathing helper with minimal logic:
      - Explore (rotate, occasional step) until each required landmark ID has
        been observed at least once (not necessarily at the same time).
      - Then, stabilize briefly and move toward the midpoint.

    Sign convention note:
      - Robot API:  robot.turn_angle(+deg)  => RIGHT (clockwise)
      - Math space: +radians                => CCW (left)
    """

    def __init__(self, robot, camera, required_landmarks, step_cm=20.0, rotation_deg=20.0):
        self.robot = robot
        self.camera = camera
        self.required_landmarks = set(required_landmarks)

        self.step_cm = float(step_cm)          # default forward step per call (cm)
        self.rotation_deg = float(rotation_deg)

        self.observed_landmarks = set()
        self.all_seen = False

        # Tiny, local tunables
        self.ALIGN_DEADBAND_RAD = math.radians(5.0)   # ignore tiny heading errors
        self.MAX_TURN_STEP_RAD  = math.radians(15.0)  # cap per-call turn amount
        self.CENTER_TOL_CM      = 5.0                 # "reached" tolerance
        self.MIN_NIBBLE_CM      = 2.0                 # ensures small forward motion

    # ----------------------------- Exploration -----------------------------
    def explore_step(self, drive=False, min_dist=400):
        """
        Spin/step to find landmarks. Optionally drive forward while avoiding
        obstacles via proximity sensors.

        Returns:
            (distance_cm, angle_rad_math)
        where angle_rad_math is the heading change in mathematical convention
        (CCW positive). Distance is forward movement in cm.
        """
        dist_cm = 0.0
        angle_rad_math = 0.0

        if self.all_seen:
            return 0.0, 0.0

        if not drive:
            # Rotate in place to scan.
            # Robot +deg => CW, so math angle is NEGATIVE of that.
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
                # Turn robot to the RIGHT (positive degrees) => math negative
                self.robot.turn_angle(45)
                angle_rad_math = -math.radians(45)
            else:
                # Turn robot to the LEFT (negative degrees) => math positive
                self.robot.turn_angle(-45)
                angle_rad_math = +math.radians(45)

            self.robot.drive_distance_cm(dist_cm)

        # Update which landmarks we’ve seen (we only need each at least once)
        frame = self.camera.get_next_frame()
        objectIDs, dists, angles = self.camera.detect_aruco_objects(frame)
        if objectIDs is not None:
            self.observed_landmarks.update(objectIDs)

        self.all_seen = self.required_landmarks.issubset(self.observed_landmarks)
        return dist_cm, angle_rad_math

    def seen_all_landmarks(self):
        """True if all required landmarks have been observed at least once."""
        return self.all_seen

    # ------------------------- Go-to-midpoint step -------------------------
    def move_towards_goal_step(self, est_pose, center, step_cm=None):
        """
        Rotate gently toward the goal, then take a modest forward step.

        Returns:
            (distance_cm, angle_rad_math_applied)
        """
        if step_cm is None:
            step_cm = self.step_cm

        # Current pose (cm, rad) and goal (cm)
        rx, ry = float(est_pose.getX()), float(est_pose.getY())
        rth    = float(est_pose.getTheta())
        gx, gy = float(center[0]), float(center[1])

        dx, dy = gx - rx, gy - ry
        dist_to_goal = float(math.hypot(dx, dy))

        # Close enough? (no motion)
        if dist_to_goal < self.CENTER_TOL_CM:
            print("reached center")
            return 0.0, 0.0

        # Heading error in math convention (CCW positive)
        heading_error = math.atan2(dy, dx) - rth
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))  # wrap [-pi, pi]

        applied_turn_math = 0.0

        # If misaligned beyond deadband, rotate a small, capped amount first
        if abs(heading_error) > self.ALIGN_DEADBAND_RAD:
            # Choose a small turn toward reducing the error (in math space)
            turn_step_math = max(-self.MAX_TURN_STEP_RAD, min(self.MAX_TURN_STEP_RAD, heading_error))
            # Convert to robot degrees: robot +deg => CW => math negative
            turn_deg_cmd = -math.degrees(turn_step_math)
            self.robot.turn_angle(turn_deg_cmd)
            applied_turn_math = turn_step_math
        else:
            applied_turn_math = 0.0

        # Drive forward a modest step toward the goal (ensure a tiny nibble)
        move_distance = float(min(step_cm, dist_to_goal))
        move_distance = max(self.MIN_NIBBLE_CM, move_distance)
        self.robot.drive_distance_cm(move_distance)

        # Return exactly what we commanded, in math convention
        return move_distance, applied_turn_math
