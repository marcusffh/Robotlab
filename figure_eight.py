import time
import robot

arlo = robot.Robot()

# ---- per-robot calibration (measure once, tweak here) ----
CAL_KL = 0.970   # left wheel scale
CAL_KR = 1.000   # right wheel scale (e.g., 0.985 means right is a bit “strong”)
MIN_PWR = 40     # per robot.py: avoid <40 except 0; valid 30..127, recommended ≥40
MAX_PWR = 127

#How do i calculate the relationship between wheel power and drive time , to go exactly 3/4 of a perfect circle?
# Make it Run in a circle, until you are happy with the size of the circle
# Adjust the time so it runs 3/4 of the circle
# ? calculate the ratio between wheel power and circle size
# Optionally make a function where you can adjust the size of the figure eight








## Circle test
arlo.do_diff(127 * CAL_KL, 60 * CAL_KR, 1, 1)
time.sleep(2)
arlo.stop()

"""
#Test the parameters until it moves in a figure eight
# go 3/4 of a circle
arlo.do_diff(127 * CAL_KL, 60 * CAL_KR, 1, 1)
time.sleep(2)
arlo.stop()
time.sleep(0.2)  # small settle

# go straight
arlo.do_diff(127 * CAL_KL, 127 * CAL_KR, 1, 1)
time.sleep(1)
arlo.stop()
time.sleep(0.2)  # small settle

# go 3/4 of a circle
arlo.do_diff(127 * CAL_KL, 60 * CAL_KR, 1, 1)
time.sleep(2)
arlo.stop()
time.sleep(0.2)  # small settle

# go straight
arlo.do_diff(127 * CAL_KL, 127 * CAL_KR, 1, 1)
time.sleep(1)
arlo.stop()
time.sleep(0.2)  # small settle
"""