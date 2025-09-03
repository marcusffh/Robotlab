import time
import robot

arlo = robot.Robot()

FORWARD = 1
BACKWARD = 0

speed = 64

#Uses non-blocking method
def continous_drive(duration, leftSpeed, rightSpeed, leftDir, rightDir):
    arlo.go_diff(leftSpeed, rightSpeed, leftDir, rightDir)
    start = time.perf_counter()
    elapsed = 0
    while elapsed < duration:
        elapsed = time.perf_counter() - start
        
    arlo.stop()
    time.sleep(0.2)  

for _ in range(1):  # figure-8 loops
    # curve left
    continous_drive(4, speed - 30, speed + 30, FORWARD, FORWARD)
    # curve right
    continous_drive(4, speed + 30, speed - 30, FORWARD, FORWARD)
