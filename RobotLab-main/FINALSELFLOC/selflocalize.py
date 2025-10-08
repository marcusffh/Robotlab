# Self-localization with MCL (Particle Filter) for Arlo
# Uses two known landmarks at (0,0) and (300,0) [cm], IDs default {6,7}.
# Robust to initial pose anywhere around the boxes (north/south), handles alien IDs, and performs spin-scan recovery.
# Noah/REX — Oct 2025

import math
import random
import time
from collections import deque

import numpy as np

# --- Project dependencies (works with your repo structure) ---
try:
    import camera  # expected to provide Camera(camidx, ...) with .getObservations()
except ImportError:
    raise ImportError("camera.py not found. Use the exercise camera module from the self-localization folder.")

CalibratedRobot = None
try:
    from RobotUtils.CalibratedRobot import CalibratedRobot  # type: ignore
except Exception:
    pass

try:
    from robot import Robot
except Exception:
    Robot = None


# ---------------- Configuration ----------------

LANDMARKS_CM = { 6: (0.0, 0.0), 7: (300.0, 0.0) }
VALID_IDS = set(LANDMARKS_CM.keys())

N_PART = 1200
PRIOR_RECT = (-150.0, 450.0, -250.0, 250.0)  # xmin, xmax, ymin, ymax [cm]
PRIOR_THETA = (-math.pi, math.pi)

NOISE_TRANS = 2.5
NOISE_ROT = math.radians(3)

SIGMA_D = 6.0
SIGMA_PHI = math.radians(5)

RANDOM_INJECT_FRAC = 0.05
QUALITY_WINDOW = 20
QUALITY_DROP = 0.35

SCAN_ROT_SPEED = 52
DRIVE_POWER = 58
CMD_STEP_SEC = 0.25

GOAL_XY = (150.0, 0.0)
GOAL_RADIUS = 15.0
MAX_HEADING_ERR = math.radians(7)


def ang_wrap(a):
    a = (a + math.pi) % (2*math.pi) - math.pi
    return a

def bearing_robot_to_point(px, py, theta, tx, ty):
    dx, dy = (tx - px), (ty - py)
    phi_world = math.atan2(dy, dx)
    return ang_wrap(phi_world - theta)

def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def systematic_resample(particles, weights):
    N = len(particles)
    positions = (np.arange(N) + random.random()) / N
    indexes = np.zeros(N, dtype=int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return particles[indexes]

def estimate_pose(particles):
    xs = particles[:, 0]; ys = particles[:, 1]; thetas = particles[:, 2]
    x = np.mean(xs); y = np.mean(ys)
    c = np.mean(np.cos(thetas)); s = np.mean(np.sin(thetas))
    theta = math.atan2(s, c)
    return float(x), float(y), float(theta)

def covariance_xytheta(particles):
    return np.var(particles[:,0]), np.var(particles[:,1]), np.var(particles[:,2])

def initialize_particles(N=N_PART):
    xmin, xmax, ymin, ymax = PRIOR_RECT
    thetamin, thetamax = PRIOR_THETA
    xs = np.random.uniform(xmin, xmax, size=N)
    ys = np.random.uniform(ymin, ymax, size=N)
    thetas = np.random.uniform(thetamin, thetamax, size=N)
    return np.stack([xs, ys, thetas], axis=1)

def sample_motion(particles, v_cm, w_rad, dt):
    if dt <= 0:
        return particles
    out = particles.copy()
    dx = v_cm * dt * np.cos(out[:,2])
    dy = v_cm * dt * np.sin(out[:,2])
    dth = w_rad * dt
    out[:,0] += dx; out[:,1] += dy; out[:,2] = np.array([ang_wrap(th) for th in (out[:,2] + dth)])
    out[:,0] += np.random.randn(len(out)) * NOISE_TRANS
    out[:,1] += np.random.randn(len(out)) * NOISE_TRANS
    out[:,2] += np.random.randn(len(out)) * NOISE_ROT
    out[:,2] = np.array([ang_wrap(th) for th in out[:,2]])
    return out

def update_with_observation(particles, obs):
    valid = [o for o in obs if o.get('id') in VALID_IDS]
    if not valid:
        return particles, None, None
    det = min(valid, key=lambda o: o['distance_cm'])
    lm_id = det['id']; lm_xy = LANDMARKS_CM[lm_id]
    dM = float(det['distance_cm']); phiM = float(det['bearing_rad'])

    dx = lm_xy[0] - particles[:,0]; dy = lm_xy[1] - particles[:,1]
    d_exp = np.hypot(dx, dy)
    phi_exp = np.array([bearing_robot_to_point(px, py, th, lm_xy[0], lm_xy[1])
                        for px, py, th in particles])

    err_d = (dM - d_exp) / SIGMA_D
    err_p = np.vectorize(ang_wrap)(phiM - phi_exp) / SIGMA_PHI
    logw = -0.5*(err_d**2) - np.log(SIGMA_D) - 0.5*(err_p**2) - np.log(SIGMA_PHI)
    logw -= np.max(logw)
    w = np.exp(logw); s = np.sum(w)
    if s == 0 or not np.isfinite(s):
        return particles, None, None
    w /= s

    res = systematic_resample(particles, w)
    res[:,0] += np.random.randn(len(res)) * 0.3
    res[:,1] += np.random.randn(len(res)) * 0.3
    res[:,2] += np.random.randn(len(res)) * math.radians(0.5)
    res[:,2] = np.array([ang_wrap(th) for th in res[:,2]])

    neff = 1.0 / np.sum((w+1e-12)**2)
    quality = float(neff/len(w))
    return res, quality, lm_id

def random_inject(particles, frac=RANDOM_INJECT_FRAC):
    k = max(1, int(len(particles)*frac))
    rnd = initialize_particles(N=k)
    particles[:k] = rnd
    return particles

class Drive:
    def __init__(self):
        self.robot = None
        if CalibratedRobot is not None:
            try: self.robot = CalibratedRobot()
            except Exception: self.robot = None
        if self.robot is None and Robot is not None:
            self.robot = Robot()
        if self.robot is None:
            raise RuntimeError("No robot interface available (CalibratedRobot/Robot).")

    def rotate_left(self, power=SCAN_ROT_SPEED, secs=CMD_STEP_SEC):
        self._go_diff(power, power, 0, 1, secs)

    def rotate_right(self, power=SCAN_ROT_SPEED, secs=CMD_STEP_SEC):
        self._go_diff(power, power, 1, 0, secs)

    def forward(self, power=DRIVE_POWER, secs=CMD_STEP_SEC):
        self._go_diff(power, power, 1, 1, secs)

    def backward(self, power=DRIVE_POWER, secs=CMD_STEP_SEC):
        self._go_diff(power, power, 0, 0, secs)

    def stop(self):
        try: self.robot.stop()
        except Exception: pass

    def _go_diff(self, l, r, dir_left, dir_right, secs):
        try:
            self.robot.go_diff(l, r, dir_left, dir_right)
        except Exception:
            self.robot.go_diff(l, r, dir_left, dir_right)
        time.sleep(secs)
        self.stop()
        time.sleep(0.05)

def safe_get_observations(cam):
    dets = []
    try:
        got = cam.getObservations()
        if isinstance(got, list):
            for g in got:
                if isinstance(g, dict):
                    dets.append(g)
        elif isinstance(got, tuple) and len(got) == 3:
            ids, dists, phis = got
            for i, d, p in zip(ids, dists, phis):
                dets.append({'id': int(i), 'distance_cm': float(d), 'bearing_rad': float(p)})
    except Exception:
        pass
    clean = []
    for d in dets:
        try:
            if not np.isfinite([d['distance_cm'], d['bearing_rad']]).all():
                continue
            clean.append(d)
        except Exception:
            continue
    return clean

def spin_scan(drive: Drive, cam, max_turns=24):
    obs_all = []
    for _ in range(max_turns):
        drive.rotate_left()
        time.sleep(0.05)
        obs = safe_get_observations(cam)
        obs_all.extend(obs)
        if any(o.get('id') in VALID_IDS for o in obs):
            return obs
    return obs_all

def nudge_and_rescan(drive: Drive, cam):
    drive.forward(secs=0.35)
    return spin_scan(drive, cam, max_turns=18)

# -------- Camera creation fixes (handles your errors) --------
def create_camera():
    """
    Your camera.Camera requires 'camidx'. Pass 0.
    Also, guard against camera.__del__ referencing a missing 'cam' attribute.
    """
    # Try common signatures:
    cam = None
    tried = []
    for kwargs in (
        {"camidx": 0, "display": False},
        {"camidx": 0},
    ):
        try:
            cam = camera.Camera(**kwargs)
            break
        except TypeError as e:
            tried.append(f"{kwargs} -> {e}")
        except Exception as e:
            tried.append(f"{kwargs} -> {e}")
    if cam is None:
        # Last resort: positional camidx
        cam = camera.Camera(0)

    # Avoid AttributeError in camera.__del__ if 'cam' is unset in the PiCamera2 path
    if not hasattr(cam, "cam") or cam.cam is None:
        class _NoopCam:
            def close(self): pass
        cam.cam = _NoopCam()

    return cam

# ---------- Main loop ----------
def main():
    print("[MAIN] Self-localization with PF + Camera")
    cam = create_camera()
    drive = Drive()

    particles = initialize_particles(N_PART)
    qual_hist = deque(maxlen=QUALITY_WINDOW)
    seen_ids = set()
    last_update_time = time.time()

    while True:
        obs = safe_get_observations(cam)
        if not any(o.get('id') in VALID_IDS for o in obs):
            obs = spin_scan(drive, cam, max_turns=18)
            if not any(o.get('id') in VALID_IDS for o in obs):
                obs = nudge_and_rescan(drive, cam)

        now = time.time()
        dt = max(0.05, min(0.6, now - last_update_time))
        last_update_time = now

        v_cm = 0.0; w_rad = 0.0
        particles = sample_motion(particles, v_cm, w_rad, dt)

        particles, q, lm_id = update_with_observation(particles, obs)
        if lm_id is not None:
            seen_ids.add(lm_id)
        if q is not None:
            qual_hist.append(q)

        if len(qual_hist) == QUALITY_WINDOW:
            q_now = qual_hist[-1]; q_max = max(qual_hist)
            if q_now < QUALITY_DROP * q_max:
                particles = random_inject(particles, RANDOM_INJECT_FRAC)

        x, y, th = estimate_pose(particles)
        vx, vy, vth = covariance_xytheta(particles)

        tight_xy = (vx < 200.0 and vy < 200.0)
        if (6 in seen_ids and 7 in seen_ids) and tight_xy:
            goal = GOAL_XY
            d_goal = dist((x, y), goal)
            if d_goal <= GOAL_RADIUS:
                print(f"[GOAL] Reached center ~ ({x:.1f},{y:.1f}) cm. Stopping.")
                drive.stop()
                time.sleep(0.2)
                break

            phi_to_goal = bearing_robot_to_point(x, y, th, goal[0], goal[1])
            if abs(phi_to_goal) > MAX_HEADING_ERR:
                if phi_to_goal > 0: drive.rotate_left(secs=0.20)
                else: drive.rotate_right(secs=0.20)
                w_rad = math.copysign(math.radians(35), phi_to_goal)
                particles = sample_motion(particles, 0.0, w_rad, 0.20)
            else:
                drive.forward(secs=0.30)
                particles = sample_motion(particles, v_cm=10.0, w_rad=0.0, dt=0.30)

        time.sleep(0.02)

    drive.stop()
    print("Finished.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
