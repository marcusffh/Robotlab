# Minimal: spin → see IDs {6,7} → compute midpoint → face it → drive → stop
# Requires only camera.py and a robot class (CalibratedRobot or Robot) — no particles.

import sys, os, time, math

# --- import paths so we can run from FINALSELFLOC ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Camera & robot
import camera
try:
    from RobotUtils.CalibratedRobot import CalibratedRobot as RobotClass
except Exception:
    from RobotUtils.Robot import Robot as RobotClass  # fallback


# ====== CONFIG (tweak if needed) ======
LANDMARK_IDS = (6, 7)
CAM_INDEX = 1                 # 0 or 1 depending on your Pi setup
ROTATE_SPEED_HINT = 55        # only used for time-based fallbacks
DRIVE_SPEED_HINT = 60         # only used for time-based fallbacks

# Rough motion calibration for time-based fallbacks (only used if your API lacks distance/angle commands)
DEG_PER_SEC_AT_ROTATE = 90.0  # ~90°/s when ROTATE_SPEED_HINT=55 (adjust if needed)
CM_PER_SEC_AT_DRIVE   = 25.0  # ~25 cm/s when DRIVE_SPEED_HINT=60 (adjust if needed)
ANGLE_TOL_DEG         = 5.0
CENTER_TOL_CM         = 5.0


# ====== Small robot wrapper that adapts to whichever API your class has ======
class Drive:
    def __init__(self, robot):
        self.r = robot
        # Detect capability once
        self.has_turn_deg = any(hasattr(self.r, n) for n in ("turn_degrees", "turnDegrees", "turn_deg", "turn"))
        self.has_drive_cm = any(hasattr(self.r, n) for n in ("drive_straight_cm", "driveStraightCm", "forward_cm", "drive_cm", "forward"))
        self.has_left_right_secs = all(hasattr(self.r, n) for n in ("left", "right"))  # seconds-based

    def stop(self):
        for n in ("stop", "halt", "brake"):
            if hasattr(self.r, n):
                getattr(self.r, n)()
                return
        # last resort: try zero power if present
        for n in ("go_diff",):
            if hasattr(self.r, n):
                getattr(self.r, n)(0, 0, 1, 1)
                return

    def rotate(self, angle_rad):
        angle_deg = math.degrees(angle_rad)
        sgn = 1 if angle_deg >= 0 else -1
        angle_deg = abs(angle_deg)

        # Degree-based methods
        for n in ("turn_degrees", "turnDegrees", "turn_deg", "turn", "rotate_degrees", "rotateDegrees"):
            if hasattr(self.r, n):
                try:
                    getattr(self.r, n)(sgn * angle_deg)
                    return
                except TypeError:
                    # some APIs want (deg, speed)
                    try:
                        getattr(self.r, n)(sgn * angle_deg, ROTATE_SPEED_HINT)
                        return
                    except Exception:
                        pass

        # Seconds-based left/right
        if self.has_left_right_secs:
            dur = angle_deg / max(1e-6, DEG_PER_SEC_AT_ROTATE)
            try:
                if sgn > 0:  # + = left
                    self.r.left(ROTATE_SPEED_HINT, dur)
                else:
                    self.r.right(ROTATE_SPEED_HINT, dur)
                return
            except TypeError:
                # maybe signature is left(dur) without power
                if sgn > 0:
                    self.r.left(dur)
                else:
                    self.r.right(dur)
                return

        # As a last resort, try go_diff timing if present
        if hasattr(self.r, "go_diff"):
            dur = angle_deg / max(1e-6, DEG_PER_SEC_AT_ROTATE)
            if sgn > 0:
                self.r.go_diff(ROTATE_SPEED_HINT, ROTATE_SPEED_HINT, 0, 1)  # left
            else:
                self.r.go_diff(ROTATE_SPEED_HINT, ROTATE_SPEED_HINT, 1, 0)  # right
            time.sleep(dur)
            self.stop()
            return

        raise RuntimeError("No usable rotate() method found on your robot class.")

    def forward(self, dist_cm):
        dist_cm = max(0.0, dist_cm)
        # Distance-based methods
        for n in ("drive_straight_cm", "driveStraightCm", "forward_cm", "drive_cm"):
            if hasattr(self.r, n):
                try:
                    getattr(self.r, n)(dist_cm)
                    return
                except TypeError:
                    # maybe requires speed
                    try:
                        getattr(self.r, n)(dist_cm, DRIVE_SPEED_HINT)
                        return
                    except Exception:
                        pass
        # Generic 'forward' with cm argument?
        if hasattr(self.r, "forward"):
            try:
                self.r.forward(dist_cm)
                return
            except TypeError:
                # maybe forward(speed, secs)
                dur = dist_cm / max(1e-6, CM_PER_SEC_AT_DRIVE)
                self.r.forward(DRIVE_SPEED_HINT, dur)
                return

        # Seconds-based fallback (go_diff or left/right variants)
        dur = dist_cm / max(1e-6, CM_PER_SEC_AT_DRIVE)
        if hasattr(self.r, "go_diff"):
            self.r.go_diff(DRIVE_SPEED_HINT, DRIVE_SPEED_HINT, 1, 1)
            time.sleep(dur)
            self.stop()
            return

        if hasattr(self.r, "forward"):
            # forward(secs) without speed
            self.r.forward(dur)
            return

        raise RuntimeError("No usable forward() method found on your robot class.")


# ====== Camera helpers ======
def detect_valid(cam):
    """Return dict {id: (dist_cm, bearing_rad)} for IDs 6 and 7 if seen."""
    frame = cam.get_next_frame()
    ids, dists, angs = cam.detect_aruco_objects(frame)
    out = {}
    if ids is None:
        return out
    for i, d, a in zip(ids, dists, angs):
        if i in LANDMARK_IDS:
            # store nearest per ID if multiple
            if i not in out or d < out[i][0]:
                out[i] = (float(d), float(a))
    return out


# ====== Geometry ======
def midpoint_robot_frame(L6, L7):
    d1, a1 = L6
    d2, a2 = L7
    x1, y1 = d1 * math.cos(a1), d1 * math.sin(a1)
    x2, y2 = d2 * math.cos(a2), d2 * math.sin(a2)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


# ====== Main logic ======
def main():
    print("[MAIN] Spin → see {6,7} → drive to their midpoint → stop")

    # Robot & camera
    bot = RobotClass()
    drv = Drive(bot)
    cam = camera.Camera(CAM_INDEX, robottype='arlo', useCaptureThread=False)

    try:
        # 1) Spin until both landmarks are seen (accumulate over time)
        seen_dict = {}
        t0 = time.time()
        timeout = 60.0
        print("Spinning to detect landmarks...")
        while time.time() - t0 < timeout:
            # small left rotate, then check
            drv.rotate(math.radians(15))  # small step (~15°)
            seen = detect_valid(cam)
            seen_dict.update(seen)
            for k in seen:
                print(f"Seen {k}: dist={seen[k][0]:.1f} cm, ang={seen[k][1]:.2f} rad")
            if all(k in seen_dict for k in LANDMARK_IDS):
                print("✅ Both landmarks detected.")
                break
        else:
            raise TimeoutError("Failed to see both landmarks within timeout.")

        # 2) Midpoint in robot frame
        mx, my = midpoint_robot_frame(seen_dict[6], seen_dict[7])
        d_mid = math.hypot(mx, my)
        a_mid = math.atan2(my, mx)
        print(f"Midpoint: distance={d_mid:.1f} cm, bearing={math.degrees(a_mid):.1f}°")

        # 3) Face midpoint
        if abs(math.degrees(a_mid)) > ANGLE_TOL_DEG:
            drv.rotate(a_mid)

        # 4) Drive to midpoint (minus small tolerance so we don't overshoot)
        d_go = max(0.0, d_mid - CENTER_TOL_CM)
        if d_go > 0.0:
            drv.forward(d_go)

        drv.stop()
        print("✅ Arrived at midpoint. Stopped.")

    finally:
        try:
            drv.stop()
        except Exception:
            pass
        try:
            cam.terminateCaptureThread()
        except Exception:
            pass
        print("Program finished.")


if __name__ == "__main__":
    main()
