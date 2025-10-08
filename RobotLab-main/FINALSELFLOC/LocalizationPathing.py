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

        # Slow-rotation knobs (tune to taste)
        self.spin_step_deg = 5.0      # per sub-turn while spinning
        self.spin_dwell_s  = 0.08     # pause after each sub-turn
        self.align_step_deg = 6.0     # per sub-turn during alignment after detection
        self.align_dwell_s  = 0.06

    # ----------------------- Slow turn helper -----------------------
    def turn_angle_slow(self, total_deg, step_deg=5.0, dwell_s=0.08, sample_camera=False):
        """
        Execute a turn in small steps with brief dwells so the camera can catch tags.
        Positive total_deg means CW (robot API), negative means CCW.
        """
        if abs(total_deg) < 1e-6:
            return
        sign = 1.0 if total_deg >= 0 else -1.0
        step_deg = abs(step_deg)
        remaining = abs(total_deg)

        while remaining > 0.0:
            d = min(step_deg, remaining) * sign
            self.robot.turn_angle(d)
            remaining -= abs(d)
            time.sleep(dwell_s)
            if sample_camera:
                # “Peek” a frame so the calling code can detect tags more reliably
                try:
                    _ = self.camera.get_next_frame()
                except Exception:
                    pass

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
            # Slow spin: split rotation into small steps with dwells and camera samples
            self.turn_angle_slow(self.rotation_deg,
                                 step_deg=self.spin_step_deg,
                                 dwell_s=self.spin_dwell_s,
                                 sample_camera=True)
            angle_rad_math = -math.radians(self.rotation_deg)  # robot CW = math negative
        else:
            # Small forward step with a bias away from the closer side.
            dist_cm = float(self.step_cm)
            left, center, right = self.robot.proximity_check()

            if (left < min_dist) or (center < min_dist) or (right < min_dist):
                self.robot.stop()

            if left > right:
                # CW
                self.turn_angle_slow(45,
                                     step_deg=self.spin_step_deg,
                                     dwell_s=self.spin_dwell_s,
                                     sample_camera=True)
                angle_rad_math = -math.radians(45)
            else:
                # CCW
                self.turn_angle_slow(-45,
                                     step_deg=self.spin_step_deg,
                                     dwell_s=self.spin_dwell_s,
                                     sample_camera=True)
                angle_rad_math = +math.radians(45)

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
        1) Rotate to face the midpoint using slow, stepped turning.
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

        # Slow alignment turn: split into small degrees with brief dwells and camera sampling
        turn_deg_cmd = -math.degrees(heading_error)  # robot +deg = CW
        applied_turn_math = 0.0
        if abs(turn_deg_cmd) > 1.0:  # ignore sub-degree jitter
            self.turn_angle_slow(turn_deg_cmd,
                                 step_deg=self.align_step_deg,
                                 dwell_s=self.align_dwell_s,
                                 sample_camera=True)
            applied_turn_math = heading_error

        # Straight drive toward the goal in chunks
        move_distance = float(min(step_cm, dist_to_goal))
        if move_distance > 0.0:
            self.robot.drive_distance_cm(move_distance)

        return move_distance, applied_turn_math
