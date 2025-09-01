import time
import robot

arlo = robot.Robot()
Straight = 64
RightWheelError = 2
# Drive straight forward
print(arlo.go_diff(Straight, Straight + 2, 1, 1))   # left=64, right=64, both forward
time.sleep(3)                       # move for 3 seconds

# Stop
print(arlo.stop())
