import time
import robot

arlo = robot.Robot()

FWD_SPEED  = 64   # 0..127  (safe, not too fast)
TURN_SPEED = 64
FWD_TIME   = 2.5  # seconds ≈ 1 meter at speed ~64 (will tune)
TURN_TIME  = 0.9  # seconds ≈ 90° turn at speed ~64 (will tune)

try:
    for i in range(4):
        # 1) drive 1 meter (approx)
        arlo.go_diff(FWD_SPEED, FWD_SPEED, 1, 1)
        time.sleep(FWD_TIME)
        arlo.stop()
        time.sleep(0.2)

        # 2) turn ~90° left (in place)
        arlo.go_diff(TURN_SPEED, TURN_SPEED, 0, 1)
        time.sleep(TURN_TIME)
        arlo.stop()
        time.sleep(0.2)

finally:
    arlo.stop()
