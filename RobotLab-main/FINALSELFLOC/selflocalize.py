import sys, os, time, math, numpy as np

# Add repo paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from RobotUtils.CalibratedRobot import CalibratedRobot
except ImportError:
    from RobotUtils.Robot import Robot as CalibratedRobot
import camera


# --- CONFIG ---
LANDMARK_IDS = [6, 7]  # valid ArUco IDs
ROTATE_SPEED = 55       # motor power for spinning
DRIVE_SPEED = 60        # motor power for forward motion
ROTATE_STEP = 0.25      # seconds per rotation step
CENTER_TOLERANCE_CM = 5.0  # tolerance for stopping at midpoint


def spin_and_detect(cam, arlo, timeout=40):
    """
    Spin in place until both landmarks (6 and 7) are seen.
    Returns their measured distances and bearings.
    """
    start_time = time.time()
    seen = {}

    print("Spinning to detect landmarks...")
    while time.time() - start_time < timeout:
        # Rotate left
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
    Given two landmarks in robot frame (polar form dist, angle),
    compute midpoint coordinates in robot frame (x,y).
    """
    d1, a1 = L1
    d2, a2 = L2
    x1, y1 = d1 * math.cos(a1), d1 * math.sin(a1)
    x2, y2 = d2 * math.cos(a2), d2 * math.sin(a2)
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return mx, my


def turn_towards_angle(arlo, angle_rad):
    """Turn robot on the spot to face a given bearing angle (rad)."""
    direction = 0 if angle_rad > 0 else 1  # 0 = left, 1 = right
    power = ROTATE_SPEED
    t = abs(angle_rad) / math.radians(45) * 0.6  # scale rotation time roughly
    arlo.go_diff(power, power, 1 - direction, direction)
    time.sleep(t)
    arlo.stop()


def drive_forward(arlo, distance_cm, speed=DRIVE_SPEED):
    """Drive straight forward a given distance (approx)."""
    t = distance_cm / 25.0  # empirical: 25 cm/sec at speed=60
    arlo.go_diff(speed, speed, 1, 1)
    time.sleep(t)
    arlo.stop()


def main():
    print("[MAIN] Simple landmark navigation: spin → detect → drive to midpoint → stop")

    # Initialize robot and camera
    arlo = CalibratedRobot()
    cam = camera.Camera(1, robottype='arlo', useCaptureThread=False)

    try:
        # Step 1: Spin until both landmarks are seen
        seen = spin_and_detect(cam, arlo)

        # Step 2: Compute midpoint in robot frame
        L6 = seen[6]
        L7 = seen[7]
        mx, my = compute_midpoint_in_robot_frame(L6, L7)
        dist_to_mid = math.hypot(mx, my)
        ang_to_mid = math.atan2(my, mx)
        print(f"Midpoint relative to robot: dist={dist_to_mid:.1f} cm, angle={math.degrees(ang_to_mid):.1f}°")

        # Step 3: Turn toward midpoint
        if abs(ang_to_mid) > math.radians(5):
            turn_towards_angle(arlo, ang_to_mid)

        # Step 4: Drive straight toward midpoint
        drive_forward(arlo, dist_to_mid - CENTER_TOLERANCE_CM)

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
