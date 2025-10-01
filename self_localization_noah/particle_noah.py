#!/usr/bin/env python3
"""
particle_noah.py
- Minimal Monte Carlo Localization utilities for (x, y, theta).
- Uses commanded (v, omega) as odometry (no encoders); add Gaussian motion noise.
- Measurement model: independent Gaussian on range and bearing to known landmarks.

All angles are radians; positions in meters.
"""

import math
import random
from typing import List, Tuple
import numpy as np


# -------------------------- helpers ------------------------------------------
def angle_wrap(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# -------------------------- initialization -----------------------------------
def init_particles(N: int,
                   bounds_xy: Tuple[Tuple[float, float], Tuple[float, float]],
                   theta_range: Tuple[float, float]) -> np.ndarray:
    """
    Return array of shape (N, 3): [x, y, theta].
    bounds_xy: ((xmin, xmax), (ymin, ymax))
    theta_range: (tmin, tmax)
    """
    xs = np.random.uniform(bounds_xy[0][0], bounds_xy[0][1], size=N)
    ys = np.random.uniform(bounds_xy[1][0], bounds_xy[1][1], size=N)
    ts = np.random.uniform(theta_range[0], theta_range[1], size=N)
    return np.stack([xs, ys, ts], axis=1).astype(np.float32)


# -------------------------- motion model -------------------------------------
def predict(particles: np.ndarray,
            v: float,
            omega: float,
            dt: float,
            sigma_trans: float,
            sigma_rot: float) -> None:
    """
    Unicycle prediction using commanded (v, omega) with additive noise:
      x' = x + v*dt*cos(theta)
      y' = y + v*dt*sin(theta)
      t' = t + omega*dt
    Noise:
      translational step ~ N(0, sigma_trans * |v*dt|)
      rotational step    ~ N(0, sigma_rot   * |omega*dt| + sigma_rot * |v*dt|)
    """
    if dt <= 0.0:
        return

    x = particles[:, 0]
    y = particles[:, 1]
    th = particles[:, 2]

    # mean motion
    dx = v * dt * np.cos(th)
    dy = v * dt * np.sin(th)
    dth = omega * dt

    # noise scales
    trans_scale = sigma_trans * (abs(v * dt) + 1e-6)
    rot_scale   = sigma_rot   * (abs(omega * dt) + 0.25 * abs(v * dt) + 1e-6)

    # add noise
    dx += np.random.normal(0.0, trans_scale, size=len(particles))
    dy += np.random.normal(0.0, trans_scale, size=len(particles))
    dth += np.random.normal(0.0, rot_scale, size=len(particles))

    particles[:, 0] = x + dx
    particles[:, 1] = y + dy
    particles[:, 2] = (th + dth + np.pi) % (2.0 * np.pi) - np.pi


# -------------------------- measurement model --------------------------------
def _expected_meas(px: float, py: float, pth: float, lx: float, ly: float):
    dx = lx - px
    dy = ly - py
    r  = math.hypot(dx, dy)
    bearing = angle_wrap(math.atan2(dy, dx) - pth)
    return r, bearing


def weight(particles: np.ndarray,
           detections: List[Tuple[int, float, float]],
           landmarks: dict,
           sigma_r: float,
           sigma_b: float) -> np.ndarray:
    """
    detections: list of (id, r_meas, b_meas)
    returns weights of shape (N,), normalized (sum=1). If no detections, uniform.
    """
    N = particles.shape[0]
    if not detections:
        return np.ones(N, dtype=np.float32) / float(N)

    inv_2pi_rb = 1.0 / (2.0 * math.pi * sigma_r * sigma_b + 1e-9)
    inv_2sig_r2 = 0.5 / (sigma_r * sigma_r + 1e-12)
    inv_2sig_b2 = 0.5 / (sigma_b * sigma_b + 1e-12)

    w = np.ones(N, dtype=np.float64)
    for (lid, r_meas, b_meas) in detections:
        if lid not in landmarks:
            continue
        lx, ly = landmarks[lid]
        # vectorized expected measurement
        dx = lx - particles[:, 0]
        dy = ly - particles[:, 1]
        r  = np.hypot(dx, dy)
        b  = np.arctan2(dy, dx) - particles[:, 2]
        b  = (b + np.pi) % (2.0 * np.pi) - np.pi

        dr = r - r_meas
        db = ((b - b_meas + np.pi) % (2.0 * np.pi)) - np.pi

        # Gaussian likelihood product
        lw = np.exp(- (dr * dr) * inv_2sig_r2 - (db * db) * inv_2sig_b2) * inv_2pi_rb
        # Protect against underflow
        w *= (lw + 1e-12)

    # normalize
    s = float(np.sum(w))
    if s <= 0.0 or not np.isfinite(s):
        return np.ones(N, dtype=np.float32) / float(N)
    return (w / s).astype(np.float32)


# -------------------------- resampling ---------------------------------------
def resample_systematic(particles: np.ndarray,
                        weights: np.ndarray,
                        recovery_frac: float = 0.05,
                        bounds_xy: Tuple[Tuple[float, float], Tuple[float, float]] = ((-1, 4), (-1, 4)),
                        theta_range: Tuple[float, float] = (-math.pi, math.pi)) -> np.ndarray:
    """
    Low-variance (systematic) resampling. Adds 'recovery_frac' fresh random particles.
    """
    N = len(particles)
    M = int((1.0 - recovery_frac) * N)
    if M < 1:
        M = N

    # systematic resampling for M
    positions = (np.arange(M) + random.random()) / M
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # avoid edge issues
    idx = np.searchsorted(cumsum, positions)

    newp = particles[idx].copy()

    # recovery particles
    R = N - M
    if R > 0:
        rec = init_particles(R, bounds_xy, theta_range)
        newp = np.vstack([newp, rec])

    return newp


# -------------------------- estimate -----------------------------------------
def effective_sample_size(weights: np.ndarray) -> float:
    """ESS = 1 / sum(w_i^2)."""
    return 1.0 / float(np.sum(weights * weights) + 1e-12)


def estimate_pose(particles: np.ndarray, weights: np.ndarray):
    """
    Weighted mean of x,y and circular mean of theta.
    Returns (x, y, theta).
    """
    w = weights.reshape(-1, 1)
    xy = np.sum(particles[:, :2] * w, axis=0)  # (2,)
    c = np.sum(np.cos(particles[:, 2]) * weights)
    s = np.sum(np.sin(particles[:, 2]) * weights)
    th = math.atan2(s, c)
    return float(xy[0]), float(xy[1]), th
