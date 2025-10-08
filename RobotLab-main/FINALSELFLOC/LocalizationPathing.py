# LocalizationPathing.py
import time
import math
import numpy as np

class LocalizationPathing:
    """
    Explore until both required landmark IDs have been seen at least once,
    then rotate toward the midpoint (known world coord) and drive there.

    CRITICAL FIX:
      Use a single mapping from math angle (CCW +) to robot.turn_angle(...)
      via _robot_turn_math(). If your robot.turn_angle(+deg) = CW, set
      robot_cw_positive=True (typical Arlo). If it's CCW, set False.
    """

    def __init__(self, robot, camera, required_landmarks,
                 step_cm=15.0, rotation_deg=18.0, robot_cw_positive=True):
        self.robot = robot
        self.camera = camera
        self.required_landmarks = set(required_landmarks)
        self.step_cm = float(step_cm)
        self.rotation_deg = float(rotation_deg)
        self.robot_cw_positive = bool(robot_cw_positive)

        self.observed_landmarks = set()
        self.all_seen = False

        # Tuning (small & safe)
        self.align_deadband_rad = math.radians(6.0)   # don't over-aim
        self.max_turn_step_rad  = math.radians(12.0)  # cap per-loop rotation
        self.nibble_cm          = 6.0                 # tiny forward step after turning (breaks spin-lock)

    # ---- unified mapping: math angle (CCW +) -> robot.turn_angle(...) ----
    def _robot_turn_math(self, angle_rad):
        """
        Rotate the robot by 'angle_rad' in math convention (CCW positive).
        Handles robot API sign so physical rotation matches math.
        """
        deg = math.degrees(angle_rad)
        if abs(deg) < 1.0:
            return
        if self.robot_cw_positive:
            # robot.turn_angle(+deg) = CW/right -> invert sign for CCW math
            self.robot.turn_angle(-deg)
        else:
            # robot.turn_angle(+deg) = CCW/left -> same sign as math
            self.robot.turn_angle(+deg)

    # ----------------------------------------------------------------------

    def explore_step(self, drive=False, min_dist=400):
        """Quick spin/step to accumulate landmark IDs. Returns (distance_cm, angle_rad_applied_in_math)."""
        dist = 0.0
        angle_applied = 0.0

        if self.all_seen:
            return 0.0, 0.0

        if not drive:
            # rotate by rotation_deg (in math: + means CCW)
            angle_math = math.radians(self.rotation_deg)
            self._robot_turn_math(angle_math)
            angle_applied = angle_math
            time.sleep(0.12)  # brief dwell so camera can see
        else:
            dist = self.step_cm
            left, center, right = self.robot.proximity_check()
            if left < min_dist or center < min_dist or right < min_dist:
                self.robot.stop()
            # bias away from closer side
            if left > right:
                angle_math = math.radians(+18.0)   # CCW in math
            else:
                angle_math = math.radians(-18.0)   # CW in math
            self._robot_turn_math(angle_math)
            angle_applied = angle_math
            self.robot.drive_distance_cm(dist)

        # Update seen IDs
        frame = self.camera.get_next_frame()
        ids, dists, angs = self.camera.detect_aruco_objects(frame)
        if ids is not None:
            self.observed_landmarks.update(ids)
        self.all_seen = self.required_landmarks.issubset(self.observed_landmarks)

        return dist, angle_applied

    def seen_all_landmarks(self):
        return self.all_seen

    def move_towards_goal_step(self, est_pose, center, step_cm=None):
        """
        Rotate a capped amount toward the world midpoint (correct turn mapping),
        then ALWAYS take a tiny forward nibble. Once aligned enough, take bigger steps.
        Returns (distance_cm_commanded, angle_rad_applied_this_step_in_math).
        """
        if step_cm is None:
            step_cm = self.step_cm

        rx, ry = float(est_pose.getX()), float(est_pose.getY())
        rth    = float(est_pose.getTheta())
        gx, gy = float(center[0]), float(center[1])

        dx, dy = gx - rx, gy - ry
        dist_to_center = float(math.hypot(dx, dy))
        if dist_to_center < 12.0:  # slightly wider than final check in main
            return 0.0, 0.0

        # math heading error CCW-positive, normalized
        ang_err = math.atan2(dy, dx) - rth
        ang_err = math.atan2(math.sin(ang_err), math.cos(ang_err))

        # If misaligned, rotate only a capped amount
        if abs(ang_err) > self.align_deadband_rad:
            turn_step = max(-self.max_turn_step_rad, min(self.max_turn_step_rad, ang_err))
            self._robot_turn_math(turn_step)

            # Nibble forward to break spin-lock and move geometry
            move_distance = min(self.nibble_cm, max(0.0, dist_to_center - 2.0))
            if move_distance > 0.0:
                self.robot.drive_distance_cm(move_distance)
            return move_distance, turn_step

        # Aligned enough: take a bigger step toward goal
        move_distance = float(min(step_cm, dist_to_center))
        self.robot.drive_distance_cm(move_distance)
        return move_distance, 0.0
