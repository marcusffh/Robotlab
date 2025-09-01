import time
import robot

arlo = robot.Robot()
Straight = 64
RightWheelError = 0.98
# Drive straight forward
print(arlo.go_diff(Straight, Straight * RightWheelError , 1, 1))   # left=64, right=64, both forward
time.sleep(3)                       # move for 3 seconds

# Stop
print(arlo.stop())
