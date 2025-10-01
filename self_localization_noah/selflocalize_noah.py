# selflocalize_noah.py — runner for Exercise 5 (imports kept SIMPLE)
import time, math
from Robotutils import robot  # Robotutils/robot.py  (ensure Robotutils has __init__.py)
from self_localization_noah.particle_noah import ParticleFilterNoah, wrap_pi
from self_localization_noah.camera_noah import CameraNoah, compress_duplicates_by_id

# ---- Landmarks: set YOUR actual ArUco IDs here ----
LANDMARKS = {
    9:  (0.0,   0.0),   # left marker at (0,0) cm
    11: (300.0, 0.0),   # right marker at (300,0) cm
}
MIDPOINT = (150.0, 0.0)
BOUNDS = (-50.0, 350.0, -150.0, 150.0)

# PF noise (start here; tune if needed)
SIGMA_D   = 10.0                    # distance noise [cm]
SIGMA_PHI = math.radians(6.0)       # bearing noise [rad]
SIGMA_XY  = 1.0                     # process xy [cm/step]
SIGMA_TH  = math.radians(2.0)       # process theta [rad/step]

# Control and motor mapping
BASE_POWER = 60     # ≥ ~40 to avoid stalling motors
DIFF_GAIN  = 40     # power change per rad/s
K_TH = 1.2          # heading P-gain
K_V  = 0.8          # linear speed gain (cm/s per cm error)
V_MAX = 20.0        # cm/s
W_MAX = 1.5         # rad/s
ARRIVE_CM = 10.0

def vomega_to_diff(v_cm_s, w_rad_s):
    dl = int(BASE_POWER + DIFF_GAIN * (-w_rad_s))
    dr = int(BASE_POWER + DIFF_GAIN * ( w_rad_s))
    dl = max(0, min(127, dl))
    dr = max(0, min(127, dr))
    dirL = 1 if v_cm_s >= 0 else 0
    dirR = 1 if v_cm_s >= 0 else 0
    if BASE_POWER < 40:
        return 0, 0, 0, 0
    return dl, dr, dirL, dirR

def main():
    arlo = robot.Robot()
    cam = CameraNoah()
    pf = ParticleFilterNoah(
        N=800,
        landmarks=LANDMARKS,
        bounds=BOUNDS,
        sigma_motion_xy=SIGMA_XY,
        sigma_motion_th=SIGMA_TH,
        sigma_meas_d=SIGMA_D,
        sigma_meas_th=SIGMA_PHI,
        seed=42
    )

    seen = set()
    last = time.perf_counter()

    try:
        while True:
            dets = cam.detect()
            dets = compress_duplicates_by_id(dets)

            if dets:
                pf.update(dets)
                for d in dets:
                    seen.add(int(d["id"]))

            now = time.perf_counter()
            dt = max(1e-3, now - last)
            last = now
            pf.predict(10.0, 0.0, dt)   # gentle forward drift helps PF; set 0.0 if you prefer
            pf.resample_if_needed()

            x, y, th = pf.estimate()
            have_both = all(lid in seen for lid in LANDMARKS)

            if have_both:
                dx, dy = MIDPOINT[0] - x, MIDPOINT[1] - y
                dist = math.hypot(dx, dy)
                if dist < ARRIVE_CM:
                    print("Arrived at midpoint (estimated).")
                    arlo.stop()
                    break

                goal = math.atan2(dy, dx)
                th_err = wrap_pi(goal - th)
                v = max(0.0, min(V_MAX, K_V * dist))
                w = max(-W_MAX, min(W_MAX, K_TH * th_err))
                L, R, dL, dR = vomega_to_diff(v, w)
                arlo.go_diff(L, R, dL, dR)
            else:
                # Scan to acquire both landmarks
                L, R, dL, dR = vomega_to_diff(0.0, 0.6)
                arlo.go_diff(L, R, dL, dR)

            time.sleep(0.05)

    finally:
        try: arlo.stop()
        except: pass
        try: cam.close()
        except: pass

if __name__ == "__main__":
    main()
