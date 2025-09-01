import time
import robot

arlo = robot.Robot()

SPEED  = 64   # 0..127  (safe, not too fast)
TIME   = 2.5  # seconds ≈ 1 meter at speed ~64 (will tune)
TURN_TIME  = 0.9  # seconds ≈ 90° turn at speed ~64 (will tune)

# ---- per-robot calibration (measure once, tweak here) ----
CAL_KL = 0.970   # left wheel scale
CAL_KR = 1.000   # right wheel scale (e.g., 0.985 means right is a bit “strong”)
MIN_PWR = 40     # per robot.py: avoid <40 except 0; valid 30..127, recommended ≥40
MAX_PWR = 127

def clamp_power(p):
    return max(MIN_PWR, min(MAX_PWR, int(round(p))))

def go(speed, secs, xdir, ydir):
    # apply calibration scales so drift correction follows any speed you choose
    l = clamp_power(speed * CAL_KL) if speed > 0 else 0
    r = clamp_power(speed * CAL_KR) if speed > 0 else 0
    arlo.go_diff(l, r, xdir, ydir)
    time.sleep(secs)
    arlo.stop()
    time.sleep(0.2)  # small settle


try:
    for i in range(4):
        # 1) drive 1 meter (approx)
        go(SPEED, TIME, 1, 1)
        time.sleep(0.2)
        arlo.stop()
        time.sleep(0.2)

        # 2) turn ~90° left (in place)
        go(SPEED,TURN_TIME, 0, 1)
        time.sleep(TURN_TIME)
        arlo.stop()
        time.sleep(0.2)

finally:
    arlo.stop()
