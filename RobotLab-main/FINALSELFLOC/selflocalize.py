# Minimal self-localization: spin → see ArUco IDs {6,7} → compute midpoint → face it → drive → stop
# Depends ONLY on: camera.py and your CalibratedRobot (provided in the chat).
# No particles, no GUI.

import sys, os, time, math

# --- let us import from parent folder when launched inside FINALSELFLOC ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import camera
from RobotUtils.CalibratedRobot import CalibratedRobot  # your class (with turn_angle, drive_distance, stop)

# ===== CONFIG =====
LANDMARK_IDS = (6, 7)
CAM_INDEX_TRY = (1, 0)          # try cam 1 first, then 0 (change if you know the exact one)
ANGLE_STEP_DEG = 15.0           # spin step per detection attempt
ANGLE_TOL_DEG = 5.0             # don't micro-adjust below this
CENTER_TOL_CM = 5.0             # stop a tad early to avoid overshoot

# ===== Helpers =====
def detect_valid(cam):
    """
    Return dict {id: (dist_cm, bearing_rad)} for IDs {6,7} IF seen in the current frame.
    Keeps only the nearest measurement if multiple detections of the same ID exist.
    """
    frame = cam.get_next_frame()
    ids, dists, angs = cam.detect_aruco_objects(frame)
    out = {}
    if ids is None:
        return out
    for i, d, a in zip(ids, dists, angs):
        if i in LANDMARK_IDS:
            if i not in out or d < out[i][0]:
                out[i] = (float(d), float(a))
    return out

def midpoint_robot_frame(L6, L7):
    """Convert (d,phi) for each landmark to (x,y) and return the midpoint in robot frame."""
    d1, a1 = L6; d2, a2 = L7
    x1, y1 = d1 * math.cos(a1), d1 * math.sin(a1)
    x2, y2 = d2 * math.cos(a2), d2 * math.sin(a2)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def main():
    print("[MAIN] Minimal: spin → detect {6,7} → midpoint → turn → drive → stop")
    # Initialize camera (try indices in order)
    cam = None
    last_err = None
    for idx in CAM_INDEX_TRY:
        try:
            cam = camera.Camera(idx, robottype='arlo', useCaptureThread=False)
            print(f"[INFO] Camera opened on index {idx}")
            break
        except Exception as e:
            last_err = e
            continue
    if cam is None:
        raise RuntimeError(f"Could not open camera. Last error: {last_err}")

    bot = CalibratedRobot()  # has turn_angle(deg), drive_distance(meters), stop()

    try:
        # 1) Spin in place in small steps until both IDs have been seen (accumulate over time)
        seen = {}
        t0 = time.time(); timeout_s = 60.0
        print("[STEP] Spinning to detect landmarks 6 and 7...")
        while time.time() - t0 < timeout_s:
            # Rotate left a bit
            bot.turn_angle(+ANGLE_STEP_DEG)
            time.sleep(0.25)
            # Check detections
            curr = detect_valid(cam)
            if curr:
                seen.update(curr)
                for k, (d, a) in curr.items():
                    print(f"  - Seen {k}: dist={d:.1f} cm, angle={a:.2f} rad")
            if all(k in seen for k in LANDMARK_IDS):
                print("[OK] Both landmarks detected.")
                break
        else:
            raise TimeoutError("Failed to see both landmarks within 60s. Try adjusting lighting/IDs/camera index.")

        # 2) Compute midpoint (robot frame)
        mx, my = midpoint_robot_frame(seen[6], seen[7])
        dist_mid_cm = math.hypot(mx, my)
        ang_mid_rad = math.atan2(my, mx)
        ang_mid_deg = math.degrees(ang_mid_rad)
        print(f"[STEP] Midpoint: distance={dist_mid_cm:.1f} cm, bearing={ang_mid_deg:.1f}°")

        # 3) Turn to face midpoint (only if meaningful)
        if abs(ang_mid_deg) > ANGLE_TOL_DEG:
            print(f"[STEP] Turning {ang_mid_deg:.1f}° to face midpoint...")
            bot.turn_angle(ang_mid_deg)
            time.sleep(0.05)

        # 4) Drive straight to midpoint (stop a little early)
        go_cm = max(0.0, dist_mid_cm - CENTER_TOL_CM)
        go_m = go_cm / 100.0
        if go_m > 0.0:
            print(f"[STEP] Driving forward ~{go_cm:.1f} cm...")
            bot.drive_distance(go_m)  # meters
        bot.stop()
        print("[DONE] Arrived at midpoint between landmarks. Stopped.")

    finally:
        try:
            cam.terminateCaptureThread()
        except Exception:
            pass
        try:
            bot.stop()
        except Exception:
            pass
        print("Program finished.")

if __name__ == "__main__":
    main()
