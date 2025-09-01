import time
import robot

arlo = robot.Robot()

# Drive straight forward
print(arlo.go_diff(64, 67, 1, 1))   # left=64, right=64, both forward
time.sleep(3)                       # move for 3 seconds

# Stop
print(arlo.stop())
