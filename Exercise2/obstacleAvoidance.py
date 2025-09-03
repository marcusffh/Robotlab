# obstacle_avoidance.py
import time
from CalibratedRobot import CalibratedRobot

# Initialize the robot
calArlo = CalibratedRobot()

def drive_with_obstacle_avoidance(calArlo, total_distance=2, speed=None, min_dist=200):
    """
    Drive forward while avoiding obstacles using the front sensors.
    - total_distance: total distance in meters to attempt driving
    - min_dist: minimum distance to obstacle before reacting
    """
    if speed is None:
        speed = calArlo.default_speed

    distance_covered = 0
    step = 0.05  # small step in meters for checking sensors frequently

    while distance_covered < total_distance:
        # Read front sensors
        left = calArlo.arlo.read_left_ping_sensor()
        center = calArlo.arlo.read_front_ping_sensor()
        right = calArlo.arlo.read_right_ping_sensor()

        if center < min_dist or left < min_dist or right < min_dist:
        # Decide which way to turn
            if left > right:
                calArlo.turn_angle(60, speed)
            else:
                calArlo.turn_angle(-60, speed)


        # Drive a small step forward
        calArlo.drive_distance(step, speed=speed)
        distance_covered += step

        # Small delay to avoid overwhelming the robot
        time.sleep(0.05)

    calArlo.stop()


try:
    drive_with_obstacle_avoidance(calArlo, total_distance=5)
finally:
    calArlo.stop()
