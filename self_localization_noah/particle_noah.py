import numpy as np
import math

def wrap_pi(a):
    return (a + math.pi) % (2*math.pi) - math.pi

def _gauss_log(err, sigma):
    return -0.5*math.log(2*math.pi*sigma*sigma) - 0.5*(err/sigma)**2

class ParticleFilterNoah:
    def __init__(self, N, landmarks, bounds,
                 sigma_motion_xy=1.0, sigma_motion_th=math.radians(2.0),
                 sigma_meas_d=10.0, sigma_meas_th=math.radians(6.0),
                 seed=None):
        self.rng = np.random.default_rng(seed)
        self.N = int(N)
        self.landmarks = dict(landmarks)
        self.sxy, self.sth = float(sigma_motion_xy), float(sigma_motion_th)
        self.sd, self.sphi = float(sigma_meas_d), float(sigma_meas_th)

        xmin, xmax, ymin, ymax = bounds
        self.x  = self.rng.uniform(xmin, xmax, self.N)
        self.y  = self.rng.uniform(ymin, ymax, self.N)
        self.th = self.rng.uniform(-math.pi, math.pi, self.N)
        self.w  = np.ones(self.N)/self.N

    # motion: v,omega + Gaussian process noise (exercise baseline)
    def predict(self, v_cm_s, omega_rad_s, dt):
        self.x  += v_cm_s * dt * np.cos(self.th)
        self.y  += v_cm_s * dt * np.sin(self.th)
        self.th  = wrap_pi(self.th + omega_rad_s * dt)
        self.x  += self.rng.normal(0.0, self.sxy, self.N)
        self.y  += self.rng.normal(0.0, self.sxy, self.N)
        self.th  = wrap_pi(self.th + self.rng.normal(0.0, self.sth, self.N))

    # measurement: independent Gaussians on distance & bearing; product over landmarks
    def update(self, detections):
        if not detections:
            return

        # nearest per ID (caller already does this, but keep guard)
        best = {}
        for d in detections:
            lid = int(d["id"])
            if lid not in self.landmarks:
                continue
            dist = float(d["distance_cm"])
            phi  = float(d["phi_rad"])
            if lid not in best or dist < best[lid][0]:
                best[lid] = (dist, phi)
        if not best:
            return

        logw = np.zeros(self.N)
        cth, sth = np.cos(self.th), np.sin(self.th)
        e_th     = np.stack([cth,  sth], axis=1)
        e_th_hat = np.stack([-sth, cth], axis=1)

        for lid, (dM, phiM) in best.items():
            lx, ly = self.landmarks[lid]
            dx = lx - self.x
            dy = ly - self.y
            di = np.hypot(dx, dy) + 1e-9
            e_l = np.stack([dx/di, dy/di], axis=1)

            dot_main = np.clip((e_l * e_th).sum(axis=1), -1.0, 1.0)
            dot_orth = (e_l * e_th_hat).sum(axis=1)
            sign_term = np.sign(dot_orth)
            phi_exp = sign_term * np.arccos(dot_main)

            logw += _gauss_log(dM - di, self.sd)
            logw += _gauss_log(wrap_pi(phiM - phi_exp), self.sphi)

        logw += np.log(self.w + 1e-300)
        logw -= logw.max()
        w = np.exp(logw)
        s = w.sum()
        self.w = (w/s) if s > 0 and np.isfinite(s) else np.ones(self.N)/self.N

    def neff(self):
        return 1.0 / np.sum(self.w*self.w)

    def resample_if_needed(self, threshold_ratio=0.5):
        if self.neff() >= threshold_ratio * self.N:
            return
        positions = (self.rng.random() + np.arange(self.N)) / self.N
        c = np.cumsum(self.w)
        idx = np.searchsorted(c, positions)
        self.x, self.y, self.th = self.x[idx].copy(), self.y[idx].copy(), self.th[idx].copy()
        self.w.fill(1.0/self.N)

    def estimate(self):
        xh = float(np.sum(self.w * self.x))
        yh = float(np.sum(self.w * self.y))
        thx = float(np.sum(self.w * np.cos(self.th)))
        thy = float(np.sum(self.w * np.sin(self.th)))
        thh = math.atan2(thy, thx)
        return xh, yh, thh
