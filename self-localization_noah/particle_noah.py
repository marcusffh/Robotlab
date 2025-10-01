# particle_noah.py
import numpy as np

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def gaussian_logpdf(err, sigma):
    return -0.5*np.log(2*np.pi*sigma**2) - 0.5*(err/sigma)**2

class ParticleFilterNoah:
    def __init__(self, N, landmarks, bounds,
                 sigma_motion_xy=1.0, sigma_motion_th=np.deg2rad(2.0),
                 sigma_meas_d=10.0, sigma_meas_th=np.deg2rad(6.0),
                 seed=None):
        self.rng = np.random.default_rng(seed)
        self.N = int(N)
        self.landmarks = landmarks
        self.sxy = float(sigma_motion_xy)
        self.sth = float(sigma_motion_th)
        self.sd  = float(sigma_meas_d)
        self.sphi= float(sigma_meas_th)

        xmin, xmax, ymin, ymax = bounds
        self.x = self.rng.uniform(xmin, xmax, self.N)
        self.y = self.rng.uniform(ymin, ymax, self.N)
        self.th= self.rng.uniform(-np.pi, np.pi, self.N)
        self.w = np.ones(self.N)/self.N

    def predict(self, v_cm_s, omega_rad_s, dt_s):
        self.x += v_cm_s * dt_s * np.cos(self.th)
        self.y += v_cm_s * dt_s * np.sin(self.th)
        self.th= wrap_pi(self.th + omega_rad_s * dt_s)
        self.x += self.rng.normal(0.0, self.sxy, self.N)
        self.y += self.rng.normal(0.0, self.sxy, self.N)
        self.th= wrap_pi(self.th + self.rng.normal(0.0, self.sth, self.N))

    def update(self, detections):
        best = {}
        for d in detections:
            lid = int(d["id"])
            dist = float(d["distance_cm"])
            phi  = float(d["phi_rad"])
            if lid not in self.landmarks:
                continue
            if lid not in best or dist < best[lid][0]:
                best[lid] = (dist, phi)

        if not best:
            return

        logw = np.zeros(self.N)
        cth, sth = np.cos(self.th), np.sin(self.th)
        e_th = np.stack([cth, sth], axis=1)
        e_th_hat = np.stack([-sth, cth], axis=1)

        for lid, (dM, phiM) in best.items():
            lx, ly = self.landmarks[lid]
            dx = lx - self.x
            dy = ly - self.y
            di = np.hypot(dx, dy) + 1e-9
            e_l = np.stack([dx/di, dy/di], axis=1)
            dot_main = (e_l * e_th).sum(axis=1).clip(-1.0, 1.0)
            dot_orth = (e_l * e_th_hat).sum(axis=1)
            sign_term = np.sign(dot_orth)
            phi_expected = sign_term * np.arccos(dot_main)
            logw += gaussian_logpdf(dM - di, self.sd)
            logw += gaussian_logpdf(wrap_pi(phiM - phi_expected), self.sphi)

        logw += np.log(self.w + 1e-300)
        logw -= logw.max()
        w = np.exp(logw)
        s = w.sum()
        if s <= 0.0 or not np.isfinite(s):
            self.w[:] = 1.0/self.N
        else:
            self.w = w / s

    def neff(self):
        return 1.0 / np.sum(self.w**2)

    def resample_if_needed(self, threshold_ratio=0.5):
        if self.neff() >= threshold_ratio * self.N:
            return
        positions = (self.rng.random() + np.arange(self.N)) / self.N
        c = np.cumsum(self.w)
        idx = np.searchsorted(c, positions)
        self.x = self.x[idx].copy()
        self.y = self.y[idx].copy()
        self.th= self.th[idx].copy()
        self.w.fill(1.0/self.N)

    def estimate(self):
        xh = np.sum(self.w * self.x)
        yh = np.sum(self.w * self.y)
        thx = np.sum(self.w * np.cos(self.th))
        thy = np.sum(self.w * np.sin(self.th))
        thh = np.arctan2(thy, thx)
        return float(xh), float(yh), float(thh)
