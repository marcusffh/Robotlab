import keyboard #library that easily lets me use keyboard imput in terminal
from Robotutils.CalibratedRobot import CalibratedRobot
import time

def RobotController():
    robot = CalibratedRobot()
    speed = robot.default_speed # refers to calibrated Robot

    step_distance = 0.1 ## Meters
    step_angle = 15 # Degress

    print("Please use the arrow keys to control the robot")
    print("Press q to terminate the program")

    try:
        while True:
            if keyboard.is_pressed('q'):
                break

            #Continuous forward
            if keyboard.is_pressed('up'):
                robot.drive(speed, speed, robot.FORWARD, robot.FORWARD)
                while keyboard.is_pressed('up'):
                    time.sleep(0.05)
                robot.stop()
                # Short press
                if not keyboard.is_pressed('up'):
                    robot.drive_distance(step_distance, direction=robot.FORWARD, speed=speed)

            # Continuous backwards
            elif keyboard.is_pressed('down'):
                robot.drive(speed, speed, robot.BACKWARD, robot.BACKWARD)
                while keyboard.is_pressed('down'):
                    time.sleep(0.05)
                robot.stop()
                if not keyboard.is_pressed('down'):
                    robot.drive_distance(step_distance, direction=robot.BACKWARD, speed=speed)

            # Continuous left
            elif keyboard.is_pressed('left'):
                robot.drive(speed, speed, robot.BACKWARD, robot.FORWARD)
                while keyboard.is_pressed('left'):
                    time.sleep(0.05)
                robot.stop()
                # Short press
                if not keyboard.is_pressed('left'):
                    robot.turn_angle(step_angle, speed=speed)

            # Continuous right
            elif keyboard.is_pressed('right'):
                robot.drive(speed, speed, robot.FORWARD, robot.BACKWARD)
                while keyboard.is_pressed('right'):
                    time.sleep(0.05)
                robot.stop()
                # Short press
                if not keyboard.is_pressed('right'):
                    robot.turn_angle(-step_angle, speed=speed)
            else:
                robot.stop()
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("Afslutter...")
    finally:
        robot.stop()

if __name__ == "__main__":
    RobotController()
