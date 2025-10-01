#!/usr/bin/env python3
"""
selflocalize_noah.py
- Full loop: spin to see both landmarks -> MCL converge -> drive to midpoint.
- Uses commanded motion as odometry for PF predict.
- Keep commands gentle (small step-and-update) for stability on Arlo.

Run:
  python3 -m self_localization_noah.selflocalize_noah
"""

import math
import time
from collections import deque

# --- Your repo robot import (with fallback) ----------------------------------
try:
    from Robotutils import robot as robot_mod
except Exception:
    import robot as robot_mod  # fallback if run from top-level

import numpy as np

from .camera_noah import LandmarkCamera
from .particle_noah import (
    init_particles, predict, weight, resample_systematic,
    estimate_pose, effective_sample_size
)

# ---------------------- CONFIG (edit for your room) --------------------------
# Known landmark world poses in meters (IDs must match your ArUco)
LANDMARKS = {
    10: (0.0, 0.0),
    20: (3.0, 0.0),
}
TARGET = ((LANDMARKS[10][0] + LANDMARKS[20][0]) * 0.5,
          (LANDMARKS[10][1] + LANDMARKS[20][1]) * 0.5)

# Particle filter config
N_PARTICLES   = 500
BOUNDS_XY     = ((-0.5, 3.5), (-1.0, 1.0))   # prior area (m) – tweak to your field
THETA_RANGE   = (-math.pi, math.pi)
SIGMA_TRANS   = 0.20        # motion noise scale (m per 1 m step) – start loose
SIGMA_ROT     = 0.20        # motion noise scale for angles
SIGMA_R       = 0.10        # measurement noise: range (m)
SIGMA_B       = 0.10        # measurement noise: bearing (rad)
RECOVERY_FRAC = 0.05

# Control
SCAN_OMEGA    = 0.35        # rad/s spin speed
FWD_SPEED     = 0.25        # m/s (commanded)
TURN_SPEED    = 0.35        # rad/s (commanded)
SEG_LEN       = 0.25        # meters per forward segment before re-localizing
HEADING_KP    = 1.0         # proportional turn on heading error
DIST_TOL      = 0.15        # stop within 15 cm
MAX_SECONDS   = 240         # safety

# Timing
LOOP_HZ       = 10.0
DT            = 1.0 / LOOP_HZ


# ---------------------- small helpers ----------------------------------------
def angle_wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def turn_in_place(arlo, omega_cmd: float, seconds: float):
    """Command a pure rotation (approx), non-blocking updates every DT."""
    steps = int(max(1, seconds * LOOP_HZ))
    for _ in range(steps):
        arlo.drive(0, 0)  # ensure neutral
        # Your robot API likely has "turn" time-based; we emulate with short pulses
        if omega_cmd >= 0:
            arlo.turn(5)         # small left pulse
        else:
            arlo.turn(-5)        # small right pulse
        time.sleep(DT)


def go_straight(arlo, seconds: float):
    """Command a straight drive via small pulses."""
    steps = int(max(1, seconds * LOOP_HZ))
    for _ in range(steps):
        arlo.drive(20, 20)   # gentle forward (tweak if you have helpers for meters)
        time.sleep(DT)


def commanded_predict(particles, v, omega, dt):
    """PF prediction using commanded motion."""
    predict(particles, v=v, omega=omega, dt=dt,
            sigma_trans=SIGMA_TRANS, sigma_rot=SIGMA_ROT)


# ---------------------- main phases ------------------------------------------
def phase_scan_until_seen_both(cam, particles):
    """
    Spin slowly and run PF with only prediction+measurement,
    until both landmark IDs have been observed at least once.
    """
    print("[SCAN] Starting spin to see both landmarks ...")
    seen = set()
    last_report = time.time()

    start = time.time()
    while time.time() - start < 60.0:  # cap scan phase at 60s
        # Predict with a pure rotation command (use omega sign to turn left)
        commanded_predict(particles, v=0.0, omega=SCAN_OMEGA, dt=DT)

        dets = cam.read()
        for (lid, r, b) in dets:
            if lid in LANDMARKS:
                seen.add(lid)

        # Weight/resample even during scan to help convergence
        w = weight(particles, dets, LANDMARKS, SIGMA_R, SIGMA_B)
        ess = effective_sample_size(w)
        if ess < 0.5 * len(particles):
            particles[:] = resample_systematic(particles, w, RECOVERY_FRAC, BOUNDS_XY, THETA_RANGE)
            w = np.ones(len(particles), dtype=np.float32) / len(particles)

        # Console feedback
        now = time.time()
        if now - last_report > 1.0:
            est = estimate_pose(particles, w)
            print(f"[SCAN] Seen IDs: {sorted(seen)} | ESS={ess:.1f} | est=({est[0]:.2f},{est[1]:.2f},{est[2]:.2f})")
            last_report = now

        # stop condition
        if all(lid in seen for lid in LANDMARKS.keys()):
            print("[SCAN] Both landmarks seen. Proceeding.")
            return True

        # physically spin a little (non-blocking-ish)
        time.sleep(DT)

    print("[SCAN] Timeout without seeing both landmarks.")
    return False


def phase_drive_to_midpoint(arlo, cam, particles):
    """
    Navigate to TARGET using short turn-go-turn segments with re-localization.
    """
    print(f"[DRIVE] Target (midpoint) = {TARGET}")
    t0 = time.time()
    weights = np.ones(len(particles), dtype=np.float32) / len(particles)

    while time.time() - t0 < MAX_SECONDS:
        # Refresh localization for a short window (stand still, take measurements)
        for _ in range(int(0.5 * LOOP_HZ)):
            commanded_predict(particles, v=0.0, omega=0.0, dt=DT)
            weights = weight(particles, cam.read(), LANDMARKS, SIGMA_R, SIGMA_B)
            if effective_sample_size(weights) < 0.5 * len(particles):
                particles[:] = resample_systematic(particles, weights, RECOVERY_FRAC, BOUNDS_XY, THETA_RANGE)
                weights = np.ones(len(particles), dtype=np.float32) / len(particles)
            time.sleep(DT)

        x, y, th = estimate_pose(particles, weights)
        dx, dy = TARGET[0] - x, TARGET[1] - y
        dist = math.hypot(dx, dy)
        goal_heading = math.atan2(dy, dx)
        dth = angle_wrap(goal_heading - th)

        print(f"[DRIVE] pose=({x:.2f},{y:.2f},{th:.2f})  dist={dist:.2f}  dth={dth:.2f}")

        if dist <= DIST_TOL:
            print("[DRIVE] Reached midpoint. Stopping.")
            arlo.stop()
            return True

        # 1) Turn towards target (small steps, updating PF)
        turn_sign = 1.0 if dth >= 0 else -1.0
        turn_time = min(abs(dth) / max(TURN_SPEED, 1e-3), 1.2)  # cap to avoid long blind spins
        steps = int(max(1, turn_time * LOOP_HZ))
        for _ in range(steps):
            # command a small rotation pulse
            if turn_sign > 0:
                arlo.turn(5)
                omega_cmd = +TURN_SPEED
            else:
                arlo.turn(-5)
                omega_cmd = -TURN_SPEED

            # PF update while turning
            commanded_predict(particles, v=0.0, omega=omega_cmd, dt=DT)
            weights = weight(particles, cam.read(), LANDMARKS, SIGMA_R, SIGMA_B)
            if effective_sample_size(weights) < 0.5 * len(particles):
                particles[:] = resample_systematic(particles, weights, RECOVERY_FRAC, BOUNDS_XY, THETA_RANGE)
                weights = np.ones(len(particles), dtype=np.float32) / len(particles)
            time.sleep(DT)

        # 2) Drive a short straight segment (SEG_LEN), re-localizing as we move
        seg_time = SEG_LEN / max(FWD_SPEED, 1e-3)
        steps = int(max(1, seg_time * LOOP_HZ))
        for _ in range(steps):
            arlo.drive(20, 20)         # gentle forward pulse
            commanded_predict(particles, v=FWD_SPEED, omega=0.0, dt=DT)
            weights = weight(particles, cam.read(), LANDMARKS, SIGMA_R, SIGMA_B)
            if effective_sample_size(weights) < 0.5 * len(particles):
                particles[:] = resample_systematic(particles, weights, RECOVERY_FRAC, BOUNDS_XY, THETA_RANGE)
                weights = np.ones(len(particles), dtype=np.float32) / len(particles)
            time.sleep(DT)

    print("[DRIVE] Gave up (timeout).")
    arlo.stop()
    return False


# ---------------------- main --------------------------------------------------
def main():
    print("[MAIN] Starting self-localization (Noah edition).")
    # Robot
    arlo = robot_mod.Robot()
    # Camera
    cam = LandmarkCamera(use_mock=False)  # set True to test on laptop
    cam.open()

    # Particles
    particles = init_particles(N_PARTICLES, BOUNDS_XY, THETA_RANGE)

    try:
        ok = phase_scan_until_seen_both(cam, particles)
        if not ok:
            print("[MAIN] Could not see both landmarks. Exiting.")
            return

        ok = phase_drive_to_midpoint(arlo, cam, particles)
        if ok:
            print("[MAIN] Done. ✅")
        else:
            print("[MAIN] Not reached target. ❌")
    finally:
        cam.close()
        try:
            arlo.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
