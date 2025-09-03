import time
import CalibratedRobot

# Initialize robot
calArlo = CalibratedRobot.CalibratedRobot()

SAFE_DISTANCE = 200  # mm
speed = 64   # calibrated default speed


def drive_with_obstacle_avoidance(calArlo, speed=speed, min_dist=SAFE_DISTANCE):
    """
    Drive continuously while checking front sensors.
    Stops and steers around obstacles when too close.
    """
    # Start continuous forward motion using calibrated power
    calArlo.drive(speed, speed, calArlo.FORWARD, calArlo.FORWARD)

    while True:
         # Read front sensors
        left = calArlo.arlo.read_left_ping_sensor()
        center = calArlo.arlo.read_front_ping_sensor()
        right = calArlo.arlo.read_right_ping_sensor()

        if left < min_dist or center < min_dist or right < min_dist:
            print("Obstacle detected! Stopping and avoiding...")
            calArlo.arlo.stop()

            # Decide turn direction based on which side is freer
            if left > right:
                calArlo.turn_angle(45)   # turn left
            else:
                calArlo.turn_angle(-45)  # turn right

            calArlo.arlo.go_diff(l_power, r_power, calArlo.FORWARD, calArlo.FORWARD)

        time.sleep(0.05)
        
try:
    drive_with_obstacle_avoidance(calArlo)
finally:
    calArlo.stop()
