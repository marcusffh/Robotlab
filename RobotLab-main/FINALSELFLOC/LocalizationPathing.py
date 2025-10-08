# LocalizationPathing.py (drop-in)
import time
import math
import numpy as np

class LocalizationPathing:
    def __init__(self, robot, camera, required_landmarks, step_cm=20, rotation_deg=20):
        self.robot = robot
        self.camera = camera
        self.required_landmarks = set(required_landmarks)
        self.step_cm = float(step_cm)
        self.rotation_deg = float(rotation_deg)

        self.observed_landmarks = set()
        self.all_seen = False

        # Tuning: prevent spin-lock and dithering
        self.align_deadband_rad = math.radians(9.0)    # consider "aligned" within ~9°
        self.max_turn_step_rad  = math.radians(10.0)   # cap turn per loop to ~10°
        self.nibble_cm          = 8.0                  # small forward move after a turn

    # ---------- Exploration (unchanged, but quick) ----------
    def explore_step(self, drive=False, min_dist=400):
        dist = 0.0
        angle_rad = 0.0

        if self.all_seen:
            return 0.0, 0.0

        if not drive:
            # Arlo: +deg = CW (right) -> math radians are NEGATIVE
            self.robot.turn_angle(self.rotation_deg)
            angle_rad = -math.radians(self.rotation_deg)
        else:
            dist = self.step_cm
            left, center, right = self.robot.proximity_check()
            if left < min_dist or center < min_dist or right < min_dist:
                self.robot.stop()
            if left > right:
                self.robot.turn_angle(20)          # CW
                angle_rad = -math.radians(20)
            else:
                self.robot.turn_angle(-20)         # CCW
                angle_rad = +math.radians(20)
            self.robot.drive_distance_cm(dist)

        # Update which landmarks we've seen (accumulates)
        frame = self.camera.get_next_frame()
        objectIDs, dists, angles = self.camera.detect_aruco_objects(frame)
        if objectIDs is not None:
            self.observed_landmarks.update(objectIDs)
        self.all_seen = self.required_landmarks.issubset(self.observed_landmarks)

        return dist, angle_rad

    def seen_all_landmarks(self):
        return self.all_seen

    # ---------- Go to midpoint with turn-then-nibble ----------
    def move_towards_goal_step(self, est_pose, center, step_cm=None):
        """
        Rotate a bit (correct sign), then ALWAYS take a small forward 'nibble'.
        Once roughly aligned, drive bigger chunks toward the midpoint.
        Returns (distance_cm_commanded, angle_rad_applied_this_step).
        """
        if step_cm is None:
            step_cm = self.step_cm

        rx, ry = float(est_pose.getX()), float(est_pose.getY())
        rth    = float(est_pose.getTheta())
        gx, gy = float(center[0]), float(center[1])

        dx, dy = gx - rx, gy - ry
        dist_to_center = float(math.hypot(dx, dy))
        if dist_to_center < 5.0:
            print("reached center")
            return 0.0, 0.0

        # Math heading error (CCW +), wrap to [-pi, pi]
        angle_err = math.atan2(dy, dx) - rth
        angle_err = math.atan2(math.sin(angle_err), math.cos(angle_err))

        # If misaligned, turn a capped amount in the CORRECT direction for Arlo:
        # key: robot.turn_angle(+deg) is CW, so send -degrees(math_angle)
        if abs(angle_err) > self.align_deadband_rad:
            turn_step = max(-self.max_turn_step_rad, min(self.max_turn_step_rad, angle_err))
            self.robot.turn_angle(-math.degrees(turn_step))   # <-- sign fix
            # Nibble forward anyway to break spin-lock
            move_distance = min(self.nibble_cm, max(0.0, dist_to_center - 2.0))
            if move_distance > 0.0:
                self.robot.drive_distance_cm(move_distance)
            return move_distance, turn_step

        # If aligned enough, take a bigger step toward the goal
        move_distance = float(min(step_cm, dist_to_center))
        self.robot.drive_distance_cm(move_distance)
        return move_distance, 0.0
