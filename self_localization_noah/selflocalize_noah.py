#!/usr/bin/env python3
"""
selflocalize_noah.py  —  Exercise 5 (MCL) using CalibratedRobot
- Scan (small calibrated turns) until both landmarks are seen
- Run particle filter (predict, weight, resample, estimate)
- Drive to the midpoint between the two landmarks with short, calibrated segments
- Uses your Robotutils/Calibratedrobot.py so motion commands are in meters/degrees

Run from repo root:
  python3 -m self_localization_noah.selflocalize_noah
"""

import math
import time
import numpy as np

# --- Robot: use your calibrated wrapper -------------------------------------
from Robotutils.Calibratedrobot import CalibratedRobot

# --- Our PF + camera wrappers ------------------------------------------------
from .camera_noah import LandmarkCamera
from .particle_noah import (
    init_particles, predict, weight, resample_systematic,
    estimate_pose, effective_sample_size
)


# ---------------------- CONFIG (edit to your setup) --------------------------
# Known landmark world poses in meters (IDs must match your ArUco markers)
LANDMARKS = {
    9: (0.0, 0.0),
    11: (3.0, 0.0),
}
TARGET = ((LANDMARKS[9][0] + LANDMARKS[11][0]) * 0.5,
          (LANDMARKS[9][1] + LANDMARKS[11][1]) * 0.5)

# Particle filter config
N_PARTICLES   = 500
BOUNDS_XY     = ((-0.5, 3.5), (-1.0, 1.0))   # prior area (m)
THETA_RANGE   = (-math.pi, math.pi)
SIGMA_TRANS   = 0.20        # motion noise scale (per 1 m)
SIGMA_ROT     = 0.20        # motion angular noise scale
SIGMA_R       = 0.10        # meas noise: range (m)
SIGMA_B       = 0.10        # meas noise: bearing (rad)
RECOVERY_FRAC = 0.05

# Controller / step sizes
SCAN_STEP_DEG = 20.0        # per scan increment (deg)
TURN_SPEED    = 50          # CalibratedRobot speed units (0..127); lower = smoother
DRIVE_SPEED   = 50
DRIVE_STEP_M  = 0.25        # forward segment length
DIST_TOL      = 0.15        # stop when within 15 cm of midpoint
MAX_SECONDS   = 240         # overall safety timeout

# Timing for non-blocking PF updates between commands
LOOP_HZ       = 10.0
DT            = 1.0 / LOOP_HZ


# ---------------------- helpers ----------------------------------------------
def angle_wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def commanded_predict_turn(particles: np.ndarray, arlo: CalibratedRobot, angle_deg: float, speed: int):
    """
    Perform PF prediction for a calibrated turn by 'angle_deg' at given 'speed'.
    We model the motion as zero-translation, pure rotation over its calibrated duration.
    """
    angle_rad = math.radians(angle_deg)
    # Calibrated duration formula from your CalibratedRobot.turn_angle():
    # duration = TURN_TIME * (abs(angleDeg)/90) * (default_speed/current_speed)
    duration = arlo.TURN_TIME * (abs(angle_deg) / 90.0) * (arlo.default_speed / float(speed))
    # For prediction we can step once with omega = angle/duration (rad/s)
    omega = (angle_rad / duration) if duration > 1e-6 else 0.0
    predict(particles, v=0.0, omega=omega, dt=duration,
            sigma_trans=SIGMA_TRANS, sigma_rot=SIGMA_ROT)


def commanded_predict_drive(particles: np.ndarray, arlo: CalibratedRobot, meters: float, speed: int):
    """
    PF prediction for a calibrated straight drive by 'meters' at given 'speed'.
    """
    # Calibrated duration formula from your CalibratedRobot.drive_distance():
    # duration = TRANSLATION_TIME * meters * (default_speed/current_speed)
    duration = arlo.TRANSLATION_TIME * meters * (arlo.default_speed / float(speed))
    # v = distance / duration
    v = (meters / duration) if duration > 1e-6 else 0.0
    predict(particles, v=v, omega=0.0, dt=duration,
            sigma_trans=SIGMA_TRANS, sigma_rot=SIGMA_ROT)


def pf_update_in_place(particles: np.ndarray, cam: LandmarkCamera, steps: int = 5):
    """
    While robot is stopped, do a brief PF measurement update loop.
    """
    weights = np.ones(len(particles), dtype=np.float32) / len(particles)
    for _ in range(steps):
        # zero-motion predict just to keep time consistent
        predict(particles, v=0.0, omega=0.0, dt=DT,
                sigma_trans=SIGMA_TRANS, sigma_rot=SIGMA_ROT)
        dets = cam.read()
        weights = weight(particles, dets, LANDMARKS, SIGMA_R, SIGMA_B)
        if effective_sample_size(weights) < 0.5 * len(particles):
            particles[:] = resample_systematic(particles, weights, RECOVERY_FRAC, BOUNDS_XY, THETA_RANGE)
            weights = np.ones(len(particles), dtype=np.float32) / len(particles)
        time.sleep(DT)
    return weights


# ---------------------- phases ------------------------------------------------
def phase_scan_until_seen_both(arlo: CalibratedRobot, cam: LandmarkCamera, particles: np.ndarray) -> bool:
    """
    Scan with small calibrated turns; after each turn, PF-predict for that turn,
    then run one measurement update. Stop once both landmark IDs have been seen.
    """
    print("[SCAN] Starting calibrated scan...")
    seen = set()
    start = time.time()
    # We’ll sweep left in small increments
    while time.time() - start < 60.0:
        # 1) Turn a small step
        arlo.turn_angle(+SCAN_STEP_DEG, speed=TURN_SPEED)
        commanded_predict_turn(particles, arlo, +SCAN_STEP_DEG, speed=TURN_SPEED)

        # 2) Single measurement update after the turn
        dets = cam.read()
        for (lid, r, b) in dets:
            if lid in LANDMARKS:
                seen.add(lid)

        w = weight(particles, dets, LANDMARKS, SIGMA_R, SIGMA_B)
        ess = effective_sample_size(w)
        if ess < 0.5 * len(particles):
            particles[:] = resample_systematic(particles, w, RECOVERY_FRAC, BOUNDS_XY, THETA_RANGE)
            w = np.ones(len(particles), dtype=np.float32) / len(particles)

        est = estimate_pose(particles, w)
        print(f"[SCAN] Seen={sorted(seen)} | ESS={ess:.1f} | est=({est[0]:.2f},{est[1]:.2f},{est[2]:.2f})")

        if all(lid in seen for lid in LANDMARKS):
            print("[SCAN] Both landmarks observed. Proceeding.")
            arlo.stop()
            return True

    print("[SCAN] Timeout without seeing both landmarks.")
    arlo.stop()
    return False


def phase_drive_to_midpoint(arlo: CalibratedRobot, cam: LandmarkCamera, particles: np.ndarray) -> bool:
    """
    Navigate to TARGET: turn-to-goal then drive a short calibrated segment.
    Re-localize; repeat until within DIST_TOL.
    """
    print(f"[DRIVE] Target (midpoint) = {TARGET}")
    t0 = time.time()

    while time.time() - t0 < MAX_SECONDS:
        # Re-localize while stopped
        weights = pf_update_in_place(particles, cam, steps=int(0.5 * LOOP_HZ))

        # Current estimate
        x, y, th = estimate_pose(particles, weights)
        dx, dy = TARGET[0] - x, TARGET[1] - y
        dist = math.hypot(dx, dy)
        goal_heading = math.atan2(dy, dx)
        dth = angle_wrap(goal_heading - th)
        print(f"[DRIVE] pose=({x:.2f},{y:.2f},{th:.2f})  dist={dist:.2f}  dth={dth:.2f}")

        if dist <= DIST_TOL:
            print("[DRIVE] Reached midpoint. ✅")
            arlo.stop()
            return True

        # 1) Turn towards target (calibrated)
        turn_deg = math.degrees(dth)
        # Limit single turn to keep updates frequent
        turn_deg = max(-60.0, min(60.0, turn_deg))
        if abs(turn_deg) > 2.0:
            arlo.turn_angle(turn_deg, speed=TURN_SPEED)
            commanded_predict_turn(particles, arlo, turn_deg, speed=TURN_SPEED)
            # Quick measurement after turning
            _ = pf_update_in_place(particles, cam, steps=2)

        # 2) Drive a short, calibrated forward segment
        step_m = min(dist, DRIVE_STEP_M)
        if step_m > 0.03:
            arlo.drive_distance(step_m, direction=arlo.FORWARD, speed=DRIVE_SPEED)
            commanded_predict_drive(particles, arlo, step_m, speed=DRIVE_SPEED)
            # Quick measurement after driving
            _ = pf_update_in_place(particles, cam, steps=2)

    print("[DRIVE] Timeout. ❌")
    arlo.stop()
    return False


# ---------------------- main --------------------------------------------------
def main():
    print("[MAIN] Starting self-localization (Noah + CalibratedRobot).")
    # Robot
    arlo = CalibratedRobot()
    # Camera
    cam = LandmarkCamera(use_mock=False)  # set True for offline testing
    cam.open()

    # Particles
    particles = init_particles(N_PARTICLES, BOUNDS_XY, THETA_RANGE)

    try:
        if not phase_scan_until_seen_both(arlo, cam, particles):
            print("[MAIN] Could not see both landmarks. Exiting.")
            return

        ok = phase_drive_to_midpoint(arlo, cam, particles)
        print("[MAIN] Done. ✅" if ok else "[MAIN] Not reached target. ❌")
    finally:
        cam.close()
        try:
            arlo.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
