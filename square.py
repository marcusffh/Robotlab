import time
import robot

arlo = robot.Robot()

SPEED      = 64     # safe speed
FWD_TIME   = 2.5    # ~1 m (tune)
TURN_TIME  = 0.90   # ~90° (tune)
MIN_PWR    = 40
MAX_PWR    = 127

# Straight-line calibration only
CAL_KL = 0.970      # tweak if it drifts right when going straight
CAL_KR = 1.000

def clamp(p): 
    return max(MIN_PWR, min(MAX_PWR, int(round(p))))

def go_straight(speed, secs, forward=True):
    l = clamp(speed * CAL_KL)
    r = clamp(speed * CAL_KR)
    dirv = 1 if forward else 0
    arlo.go_diff(l, r, dirv, dirv)
    time.sleep(secs)
    arlo.stop()
    time.sleep(0.2)

def turn_in_place(speed, secs, left=True):
    # ignore straight calibration: equal magnitudes for a clean spin
    mag = clamp(speed)
    dl, dr = (0, 1) if left else (1, 0)   # left turn vs right turn
    arlo.go_diff(mag, mag, dl, dr)
    time.sleep(secs)                      # <-- no extra sleep after this
    arlo.stop()
    time.sleep(0.2)

try:
    for _ in range(4):                    # square = 4 sides
        go_straight(SPEED, FWD_TIME, forward=True)
        turn_in_place(SPEED, TURN_TIME, left=True)
finally:
    arlo.stop()
