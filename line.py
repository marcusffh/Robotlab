import time
import robot

arlo = robot.Robot()

# ---- per-robot calibration (measure once, tweak here) ----
CAL_KL = 0.970   # left wheel scale
CAL_KR = 1.000   # right wheel scale (e.g., 0.985 means right is a bit “strong”)
MIN_PWR = 40     # per robot.py: avoid <40 except 0; valid 30..127, recommended ≥40
MAX_PWR = 127

def clamp_power(p):
    return max(MIN_PWR, min(MAX_PWR, int(round(p))))

def go_straight(speed, secs, fwd=1):
    # apply calibration scales so drift correction follows any speed you choose
    l = clamp_power(speed * CAL_KL) if speed > 0 else 0
    r = clamp_power(speed * CAL_KR) if speed > 0 else 0
    arlo.go_diff(l, r, fwd, fwd)
    time.sleep(secs)
    arlo.stop()
    time.sleep(0.2)  # small settle

# ---- use anywhere ----
go_straight(64, 3)        # ~3s straight with calibration
go_straight(80, 2)        # still straight after speed change
