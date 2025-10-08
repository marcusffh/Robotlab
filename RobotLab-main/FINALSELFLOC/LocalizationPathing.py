# LocalizationPathing.py
import time
import math
import numpy as np

class LocalizationPathing:
    """
    Minimal pathing:
      - Explore until each required landmark ID has been observed at least once.
      - Then rotate gently toward midpoint and drive in capped steps.
    Robot API: turn_angle(+deg) is CW (right). Math +rad is CCW (left).
    """

    def __init__(self, robot, camera, required_landmarks, step_cm=20.0, rotation_deg=20.0):
        self.robot = robot
        self.camera = camera
        self.required_landmarks = set(required_landmarks)

        self.step_cm = float(step_cm)          # default forward step per call (cm)
        self.rotation_deg = float(rotation_deg)

        self.observed_landmarks = set()
        self.all_seen = False

        # Small tunables
        self.ALIGN_DEADBAND_RAD = math.radians(5.0)   # ignore tiny heading errors
        self.MAX_TURN_STEP_RAD  = math.radians(15.0)  # cap per-call turn
        self.MAX_STEP_FRACTION  = 0.6                 # never take >60% of remaining distance

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
            # Robot +deg => CW; math angle = negative
            self.robot.turn_angle(self.rotation_deg)
            angle_rad_math = -math.radians(self.rotation_deg)
            time.sleep(0.2)
        else:
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

    # ------------------------- Go-to-midpoint step -------------------------
    def move_towards_goal_step(self, est_pose, center, step_cm=None, center_tol_cm=10.0):
        """
        Rotate gently toward the goal, then take a modest forward step.
        Capped to prevent overshoot; never drives if already within tolerance.

        Returns: (distance_cm, angle_rad_math_applied)
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

        applied_turn_math = 0.0

        # Rotate a small, capped amount if misaligned
        if abs(heading_error) > self.ALIGN_DEADBAND_RAD:
            turn_step_math = max(-self.MAX_TURN_STEP_RAD, min(self.MAX_TURN_STEP_RAD, heading_error))
            turn_deg_cmd = -math.degrees(turn_step_math)  # robot +deg = CW
            self.robot.turn_angle(turn_deg_cmd)
            applied_turn_math = turn_step_math

            # Recompute remaining distance after the turn (no forward yet if you prefer strict rotate-then-drive)
            # Here we still allow a small forward step to keep momentum.
        
        # Forward step: cap to both step_cm and a fraction of remaining distance
        max_fractional = self.MAX_STEP_FRACTION * dist_to_goal
        move_distance = min(step_cm, max_fractional, dist_to_goal)

        # If the cap leaves a tiny remainder (< center tol), trim movement to stop inside tolerance
        if dist_to_goal - move_distance < center_tol_cm:
            move_distance = max(0.0, dist_to_goal - center_tol_cm)

        if move_distance > 0.0:
            self.robot.drive_distance_cm(move_distance)

        return float(move_distance), float(applied_turn_math)
