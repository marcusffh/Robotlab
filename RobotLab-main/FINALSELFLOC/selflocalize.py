# explore_landmarks.py
from RobotUtils.CalibratedRobot import CalibratedRobot
import camera
import time
import numpy as np
import math
import sys, os

# Add repo paths (if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- CONFIG ---
LANDMARK_IDS = [6, 7]       # valid ArUco IDs
ROTATE_SPEED = 55           # motor power for spinning
DRIVE_SPEED  = 60           # motor power for forward motion
ROTATE_STEP  = 0.25         # seconds per rotation step during search
CENTER_TOLERANCE_CM = 5.0   # tolerance for stopping at midpoint

def normalize_angle(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))

def spin_and_detect(cam, arlo, timeout=40):
    """
    Spin in place until both landmarks (6 and 7) are seen.
    Returns their measured distances and bearings {(id): (dist_cm, angle_rad)}.
    """
    start_time = time.time()
    seen = {}

    print("Spinning to detect landmarks...")
    while time.time() - start_time < timeout:
        # Rotate LEFT/CCW steadily (this pattern is your working 'left')
        arlo.go_diff(ROTATE_SPEED, ROTATE_SPEED, 0, 1)
        time.sleep(ROTATE_STEP)
        arlo.stop()
        time.sleep(0.1)

        frame = cam.get_next_frame()
        ids, dists, angs = cam.detect_aruco_objects(frame)

        if ids is None:
            continue

        for i, d, a in zip(ids, dists, angs):
            if i in LANDMARK_IDS:
                seen[i] = (float(d), float(a))
                print(f"Seen landmark {i}: dist={d:.1f} cm, angle={a:.2f} rad")

        if all(i in seen for i in LANDMARK_IDS):
            arlo.stop()
            print("✅ Both landmarks detected.")
            return seen

    arlo.stop()
    raise TimeoutError("Failed to see both landmarks within timeout.")

def compute_midpoint_in_robot_frame(L1, L2):
    """
    Given two landmarks in robot frame (polar: dist, angle),
    compute midpoint coordinates in robot frame (x,y).
    """
    d1, a1 = L1
    d2, a2 = L2
    x1, y1 = d1 * math.cos(a1), d1 * math.sin(a1)
    x2, y2 = d2 * math.cos(a2), d2 * math.sin(a2)
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return mx, my

# ----------------- FIXED: correct left/right mapping -----------------
def turn_towards_angle(arlo, angle_rad):
    """
    Turn robot on the spot to face a given bearing angle (rad).
    Positive angle_rad = CCW/left; Negative = CW/right.
    Uses the SAME wheel patterns as spin_and_detect().
    """
    angle_rad = normalize_angle(angle_rad)
    if abs(angle_rad) < math.radians(1.0):
        return

    # crude timing model: scale by 45° = 0.6s (tune if needed)
    t = abs(angle_rad) / math.radians(45.0) * 0.6

    if angle_rad > 0:
        # LEFT / CCW  -> SAME PATTERN AS spin_and_detect
        arlo.go_diff(ROTATE_SPEED, ROTATE_SPEED, 0, 1)
    else:
        # RIGHT / CW
        arlo.go_diff(ROTATE_SPEED, ROTATE_SPEED, 1, 0)

    time.sleep(t)
    arlo.stop()
# --------------------------------------------------------------------

def drive_forward(arlo, distance_cm, speed=DRIVE_SPEED):
    """Drive straight forward a given distance (approx)."""
    if distance_cm <= 0:
        return
    # empirical: ~25 cm/sec at speed=60 -> adjust if needed
    t = distance_cm / 25.0
    arlo.go_diff(speed, speed, 1, 1)
    time.sleep(t)
    arlo.stop()

def main():
    print("[MAIN] Simple landmark navigation: spin → detect → drive to midpoint → stop")

    # Initialize robot and camera
    arlo = CalibratedRobot()
    cam = camera.Camera(1, robottype='arlo', useCaptureThread=False)

    try:
        # Step 1: Spin until both landmarks are seen (order/side doesn’t matter)
        seen = spin_and_detect(cam, arlo)

        # Step 2: Compute midpoint in *robot frame*
        L6 = seen[6]
        L7 = seen[7]
        mx, my = compute_midpoint_in_robot_frame(L6, L7)
        dist_to_mid = math.hypot(mx, my)
        ang_to_mid  = normalize_angle(math.atan2(my, mx))
        print(f"Midpoint relative to robot: dist={dist_to_mid:.1f} cm, angle={math.degrees(ang_to_mid):.1f}°")

        # Step 3: Turn toward midpoint (now correct no matter where you started)
        if abs(ang_to_mid) > math.radians(3):
            turn_towards_angle(arlo, ang_to_mid)

        # Step 4: Drive straight toward midpoint and stop just short of it
        drive_forward(arlo, max(0.0, dist_to_mid - CENTER_TOLERANCE_CM))

        print("✅ Arrived at midpoint between landmarks. Stopping.")
        arlo.stop()

    finally:
        try:
            cam.terminateCaptureThread()
        except Exception:
            pass
        arlo.stop()
        print("Program finished.")

if __name__ == "__main__":
    main()
