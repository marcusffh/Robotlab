import time
import robot

arlo = robot.Robot()

FORWARD = 1
BACKWARD = 0

CAL_KL = 0.980  
CAL_KR = 1.000   
MIN_PWR = 40   
MAX_PWR = 127

speed = 86

def clamp_power(p):
    return max(MIN_PWR, min(MAX_PWR, int(round(p))))

#Uses non-blocking method
def continous_drive(duration, leftSpeed, rightSpeed, leftDir, rightDir):
    l = clamp_power(leftSpeed * CAL_KL) if speed > 0 else 0
    r = clamp_power(rightSpeed * CAL_KR) if speed > 0 else 0
    arlo.go_diff(l, r, leftDir, rightDir)
    start = time.perf_counter()
    elapsed = 0
    while elapsed < duration:
        elapsed = time.perf_counter() - start
         

for _ in range(3):  # figure-8 loops
    # curve left
    continous_drive(6.5, speed - 30, speed + 30, FORWARD, FORWARD)
    # curve right
    continous_drive(6.45, speed + 40, speed - 20, FORWARD, FORWARD)

arlo.stop()
time.sleep(0.2) 
