# selflocalize_noah.py
import time, math
import robot
from particle_noah import ParticleFilterNoah, wrap_pi
from camera_noah import compress_duplicates_by_id

LANDMARKS = {
    9: (0.0, 0.0),
    11: (300.0, 0.0),
}
MIDPOINT = (150.0, 0.0)
BOUNDS = (-50, 350, -150, 150)

def vomega_to_diff_drive(v_cm_s, w_rad_s):
    BASE_POWER = 60
    DIFF_GAIN = 40
    dl = int(BASE_POWER + DIFF_GAIN * (-w_rad_s))
    dr = int(BASE_POWER + DIFF_GAIN * ( w_rad_s))
    dl = max(0, min(127, dl))
    dr = max(0, min(127, dr))
    dirL = 1 if v_cm_s >= 0 else 0
    dirR = 1 if v_cm_s >= 0 else 0
    if BASE_POWER < 40:
        dl = dr = 0
    return dl, dr, dirL, dirR

def main():
    arlo = robot.Robot()
    pf = ParticleFilterNoah(
        N=800,
        landmarks=LANDMARKS,
        bounds=BOUNDS,
        sigma_motion_xy=1.0,
        sigma_motion_th=math.radians(2.0),
        sigma_meas_d=10.0,
        sigma_meas_th=math.radians(6.0),
        seed=42
    )

    from camera import Camera
    cam = Camera()

    seen_ids = set()
    last_t = time.perf_counter()

    try:
        while True:
            detections = cam.detect()
            detections = compress_duplicates_by_id(detections)

            if detections:
                pf.update(detections)
                for d in detections:
                    seen_ids.add(int(d["id"]))

            now = time.perf_counter()
            dt = max(1e-3, now - last_t)
            last_t = now
            pf.predict(10.0, 0.0, dt)
            pf.resample_if_needed()

            xh, yh, thh = pf.estimate()
            have_both = all(lid in seen_ids for lid in LANDMARKS.keys())
            gx, gy = MIDPOINT
            dx, dy = gx - xh, gy - yh
            dist = math.hypot(dx, dy)
            heading_to_goal = math.atan2(dy, dx)
            th_err = wrap_pi(heading_to_goal - thh)

            if have_both and dist < 10.0:
                print(arlo.stop())
                print("Arrived at midpoint.")
                break

            if have_both:
                v = max(0.0, min(20.0, 0.8 * dist))
                w = max(-1.5, min(1.5, 1.2 * th_err))
                L, R, dL, dR = vomega_to_diff_drive(v, w)
                arlo.go_diff(L, R, dL, dR)
            else:
                L, R, dL, dR = vomega_to_diff_drive(0.0, 0.6)
                arlo.go_diff(L, R, dL, dR)

            time.sleep(0.05)
    finally:
        try:
            arlo.stop()
        except:
            pass

if __name__ == "__main__":
    main()
