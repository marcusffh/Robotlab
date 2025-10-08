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
    # Exercise camera module (returns distance + bearing to ArUco)
    import camera  # expected to provide Camera() with .getObservations()
except ImportError:
    raise ImportError("camera.py not found. Use the exercise camera module from the self-localization folder.")

# Prefer the calibrated high-level robot wrapper if you have it
CalibratedRobot = None
try:
    from RobotUtils.CalibratedRobot import CalibratedRobot  # type: ignore
except Exception:
    pass

try:
    # Fallback to low-level Robot API (go_diff etc.)
    from robot import Robot
except Exception:
    Robot = None


# ---------------- Configuration ----------------

# Landmark IDs and world positions (cm)
LANDMARKS_CM = {
    6: (0.0, 0.0),
    7: (300.0, 0.0),
}
VALID_IDS = set(LANDMARKS_CM.keys())

# Particle filter params
N_PART = 1200                 # number of particles (adjust to Pi performance)
PRIOR_RECT = (-150.0, 450.0, -250.0, 250.0)  # xmin, xmax, ymin, ymax [cm]; covers north & south of the boxes
PRIOR_THETA = (-math.pi, math.pi)

# Motion noise (std dev)
NOISE_TRANS = 2.5             # cm, per step
NOISE_ROT = math.radians(3)   # rad, per step

# Measurement noise (std dev)
SIGMA_D = 6.0                 # cm
SIGMA_PHI = math.radians(5)   # rad

# Resampling & recovery
RANDOM_INJECT_FRAC = 0.05     # fraction of particles replaced with randoms when quality drops
QUALITY_WINDOW = 20           # moving average window for weight quality
QUALITY_DROP = 0.35           # inject when quality drops below fraction of recent max

# Control
SCAN_ROT_SPEED = 52           # motor power for rotating during scan (40..90)
DRIVE_POWER = 58              # forward drive power
CMD_STEP_SEC = 0.25           # time quantum for motion commands

# Goal logic
GOAL_XY = (150.0, 0.0)        # center point between landmarks
GOAL_RADIUS = 15.0            # cm
MAX_HEADING_ERR = math.radians(7)

# ---------- Utilities ----------

def ang_wrap(a):
    """Wrap angle to [-pi, pi]."""
    a = (a + math.pi) % (2*math.pi) - math.pi
    return a

def bearing_robot_to_point(px, py, theta, tx, ty):
    """Bearing (rad) from robot pose to target point in robot frame."""
    dx, dy = (tx - px), (ty - py)
    # Angle of target in world frame
    phi_world = math.atan2(dy, dx)
    # Bearing in robot frame = phi_world - theta (wrap)
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
    """Return mean pose (x,y,theta) with angle via vector mean."""
    xs = particles[:, 0]
    ys = particles[:, 1]
    thetas = particles[:, 2]
    x = np.mean(xs)
    y = np.mean(ys)
    c = np.mean(np.cos(thetas))
    s = np.mean(np.sin(thetas))
    theta = math.atan2(s, c)
    return float(x), float(y), float(theta)

def covariance_xytheta(particles):
    """Simple covariance diag for health checks."""
    return np.var(particles[:,0]), np.var(particles[:,1]), np.var(particles[:,2])

def initialize_particles(N=N_PART):
    xmin, xmax, ymin, ymax = PRIOR_RECT
    thetamin, thetamax = PRIOR_THETA
    xs = np.random.uniform(xmin, xmax, size=N)
    ys = np.random.uniform(ymin, ymax, size=N)
    thetas = np.random.uniform(thetamin, thetamax, size=N)
    return np.stack([xs, ys, thetas], axis=1)

# ---------- Motion & Measurement Models (per exercise PDFs) ----------

def sample_motion(particles, v_cm, w_rad, dt):
    """Rotate-translate-rotate approximation with Gaussian noise (exercise model)."""
    if dt <= 0:
        return particles
    out = particles.copy()
    # Simple unicycle: integrate small step in world frame
    # Predict pose (deterministic)
    dx = v_cm * dt * np.cos(out[:,2])
    dy = v_cm * dt * np.sin(out[:,2])
    dth = w_rad * dt
    out[:,0] += dx
    out[:,1] += dy
    out[:,2] = np.array([ang_wrap(th) for th in (out[:,2] + dth)])

    # Add Gaussian noise
    out[:,0] += np.random.randn(len(out)) * NOISE_TRANS
    out[:,1] += np.random.randn(len(out)) * NOISE_TRANS
    out[:,2] += np.random.randn(len(out)) * NOISE_ROT
    out[:,2] = np.array([ang_wrap(th) for th in out[:,2]])
    return out

def likelihood_distance(d_meas, d_exp):
    # N(d_meas | d_exp, SIGMA_D^2)
    return math.exp(-0.5 * ((d_meas - d_exp)/SIGMA_D)**2) / (SIGMA_D * math.sqrt(2*math.pi))

def likelihood_bearing(phi_meas, phi_exp):
    # N(phi_meas | phi_exp, SIGMA_PHI^2), wrapping the error
    err = ang_wrap(phi_meas - phi_exp)
    return math.exp(-0.5 * (err/SIGMA_PHI)**2) / (SIGMA_PHI * math.sqrt(2*math.pi))

def update_with_observation(particles, obs):
    """
    obs: list of detections from camera:
      each item like {'id': int, 'distance_cm': float, 'bearing_rad': float}
    We use ONLY the first valid detection (closest landmark) for a clean single-landmark likelihood,
    but you can extend to product over all valid detections if desired.
    """
    valid = [o for o in obs if o.get('id') in VALID_IDS]
    if not valid:
        return particles, None, None  # no update
    # Choose closest detection (strongest constraint)
    det = min(valid, key=lambda o: o['distance_cm'])
    lm_id = det['id']
    lm_xy = LANDMARKS_CM[lm_id]
    dM = float(det['distance_cm'])
    phiM = float(det['bearing_rad'])

    # Compute expected measurements for every particle
    dx = lm_xy[0] - particles[:,0]
    dy = lm_xy[1] - particles[:,1]
    d_exp = np.hypot(dx, dy)
    phi_exp = np.array([bearing_robot_to_point(px, py, th, lm_xy[0], lm_xy[1])
                        for px, py, th in particles])

    # Likelihood per particle
    # To avoid underflow, work in log and then exp- max
    logw = -0.5*((dM - d_exp)/SIGMA_D)**2 - np.log(SIGMA_D) \
           -0.5*((np.vectorize(ang_wrap)(phiM - phi_exp))/SIGMA_PHI)**2 - np.log(SIGMA_PHI)
    # Normalize
    logw -= np.max(logw)
    w = np.exp(logw)
    s = np.sum(w)
    if s == 0 or not np.isfinite(s):
        # Degenerate — keep particles but report no update quality
        return particles, None, None

    w /= s
    # Resample
    res = systematic_resample(particles, w)
    # (Optional) small jitter after resample to avoid impoverishment
    res[:,0] += np.random.randn(len(res)) * 0.3
    res[:,1] += np.random.randn(len(res)) * 0.3
    res[:,2] += np.random.randn(len(res)) * math.radians(0.5)
    res[:,2] = np.array([ang_wrap(th) for th in res[:,2]])

    # Return resampled set and a scalar quality (effective N proxy)
    neff = 1.0 / np.sum((w+1e-12)**2)
    quality = float(neff/len(w))
    return res, quality, lm_id

def random_inject(particles, frac=RANDOM_INJECT_FRAC):
    """Replace a small fraction of worst particles with random prior samples."""
    k = max(1, int(len(particles)*frac))
    rnd = initialize_particles(N=k)
    # Replace the first k (we already resampled, so order is arbitrary)
    particles[:k] = rnd
    return particles

# ---------- Robot motion helpers ----------

class Drive:
    def __init__(self):
        self.robot = None
        if CalibratedRobot is not None:
            try:
                self.robot = CalibratedRobot()
            except Exception:
                self.robot = None
        if self.robot is None and Robot is not None:
            self.robot = Robot()
        if self.robot is None:
            raise RuntimeError("No robot interface available (CalibratedRobot/Robot).")

    # Low-level wrappers (timed)
    def rotate_left(self, power=SCAN_ROT_SPEED, secs=CMD_STEP_SEC):
        self._go_diff(power, power, dir_left=0, dir_right=1, secs=secs)

    def rotate_right(self, power=SCAN_ROT_SPEED, secs=CMD_STEP_SEC):
        self._go_diff(power, power, dir_left=1, dir_right=0, secs=secs)

    def forward(self, power=DRIVE_POWER, secs=CMD_STEP_SEC):
        self._go_diff(power, power, dir_left=1, dir_right=1, secs=secs)

    def backward(self, power=DRIVE_POWER, secs=CMD_STEP_SEC):
        self._go_diff(power, power, dir_left=0, dir_right=0, secs=secs)

    def stop(self):
        try:
            self.robot.stop()
        except Exception:
            pass

    def _go_diff(self, l, r, dir_left, dir_right, secs):
        # CalibratedRobot may have different API; fall back to go_diff if available
        try:
            self.robot.go_diff(l, r, dir_left, dir_right)
        except Exception:
            # If CalibratedRobot exposes helpers, you can add them here.
            self.robot.go_diff(l, r, dir_left, dir_right)
        time.sleep(secs)
        self.stop()
        time.sleep(0.05)

# ---------- Spin-scan + nudge recovery ----------

def spin_scan(drive: Drive, cam, max_turns=24):
    """Rotate in place, capturing observations; return latest list."""
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
    # small forward or backward move, then scan again
    drive.forward(secs=0.35)
    return spin_scan(drive, cam, max_turns=18)

def safe_get_observations(cam):
    """
    Expect camera.Camera.getObservations() -> list of dicts:
    [{'id': 6, 'distance_cm': 180.0, 'bearing_rad': +0.12}, ...]
    If your camera returns another shape, adapt here.
    """
    dets = []
    try:
        # The exercise camera typically returns (ids, dists_cm, bearings_rad) or similar.
        got = cam.getObservations()
        # Try to normalize a few known formats:
        if isinstance(got, list):
            # Assume already list of dicts
            for g in got:
                if isinstance(g, dict):
                    dets.append(g)
        elif isinstance(got, tuple) and len(got) == 3:
            ids, dists, phis = got
            for i, d, p in zip(ids, dists, phis):
                dets.append({'id': int(i), 'distance_cm': float(d), 'bearing_rad': float(p)})
    except Exception:
        pass
    # Filter NaNs and aliens
    clean = []
    for d in dets:
        try:
            if not np.isfinite([d['distance_cm'], d['bearing_rad']]).all():
                continue
            clean.append(d)
        except Exception:
            continue
    return clean

# ---------- Main loop ----------

def main():
    print("[MAIN] Self-localization with PF + Camera")
    # Camera
    cam = camera.Camera()  # must be the exercise camera class
    # Robot
    drive = Drive()

    # Particles + quality tracking
    particles = initialize_particles(N_PART)
    qual_hist = deque(maxlen=QUALITY_WINDOW)
    seen_ids = set()

    last_update_time = time.time()

    while True:
        # 1) Try to see landmarks (spin-scan if needed)
        obs = safe_get_observations(cam)
        if not any(o.get('id') in VALID_IDS for o in obs):
            obs = spin_scan(drive, cam, max_turns=18)
            if not any(o.get('id') in VALID_IDS for o in obs):
                obs = nudge_and_rescan(drive, cam)

        # 2) Motion update: we approximate commanded motion by the last action
        # Here we step the filter even when stationary to model time passing (small noise)
        now = time.time()
        dt = max(0.05, min(0.6, now - last_update_time))
        last_update_time = now

        # Our low-level loop issues short rotate/forward commands already;
        # For the filter we assume a small forward bias near zero unless we actually drive to goal later.
        v_cm = 0.0
        w_rad = 0.0
        particles = sample_motion(particles, v_cm, w_rad, dt)

        # 3) Measurement update if any valid observation
        particles, q, lm_id = update_with_observation(particles, obs)
        if lm_id is not None:
            seen_ids.add(lm_id)
        if q is not None:
            qual_hist.append(q)

        # 4) Adaptive random injection if quality dropped (kidnapped / wrong mode)
        if len(qual_hist) == QUALITY_WINDOW:
            q_now = qual_hist[-1]
            q_max = max(qual_hist)
            if q_now < QUALITY_DROP * q_max:
                particles = random_inject(particles, RANDOM_INJECT_FRAC)

        # 5) Pose estimate and goal behavior
        x, y, th = estimate_pose(particles)
        vx, vy, vth = covariance_xytheta(particles)

        # If we’ve seen both landmarks and the estimate is tight enough, go to the midpoint
        tight_xy = (vx < 200.0 and vy < 200.0)  # ~ < ~14 cm std
        if (6 in seen_ids and 7 in seen_ids) and tight_xy:
            goal = GOAL_XY
            # Check distance to goal
            d_goal = dist((x, y), goal)
            if d_goal <= GOAL_RADIUS:
                print(f"[GOAL] Reached center ~ ({x:.1f},{y:.1f}) cm. Stopping.")
                drive.stop()
                time.sleep(0.2)
                break

            # Turn towards goal if heading error large, else go forward a step
            phi_to_goal = bearing_robot_to_point(x, y, th, goal[0], goal[1])
            if abs(phi_to_goal) > MAX_HEADING_ERR:
                # rotate in place towards sign of phi
                if phi_to_goal > 0:
                    drive.rotate_left(secs=0.20)
                else:
                    drive.rotate_right(secs=0.20)
                # Model some commanded rotation in filter:
                w_rad = math.copysign(math.radians(35), phi_to_goal)
                particles = sample_motion(particles, 0.0, w_rad, 0.20)
            else:
                # advance a bit
                drive.forward(secs=0.30)
                particles = sample_motion(particles, v_cm=10.0, w_rad=0.0, dt=0.30)

        # Small idle to be gentle on CPU
        time.sleep(0.02)

    drive.stop()
    print("Finished.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
