import time
import robot

arlo = robot.Robot()

# ---- per-robot calibration (measure once, tweak here) ----
CAL_KL = 0.970   # left wheel scale
CAL_KR = 1.000   # right wheel scale (e.g., 0.985 means right is a bit “strong”)
MIN_PWR = 40     # per robot.py: avoid <40 except 0; valid 30..127, recommended ≥40
MAX_PWR = 127

#How do i calculate the relationship between wheel power and drive time , to go in a perfect circle?
# Make it Run in a circle, until you are happy with the size of the circle
# ? calculate the ratio between wheel power and circle size
# Optionally make a function where you can adjust the size of the figure eight



## What is the relationship bewteen the relative wheel power and drivetime, to complete one rotation in a cirle





#Test the parameters until it moves in a figure eight
# go a Circle
arlo.do_diff(127 * CAL_KL, 60 * CAL_KR, 1, 1)
time.sleep(2)

# A circle 
arlo.do_diff(60 * CAL_KL, 127 * CAL_KR, 1, 1)
time.sleep(2)


