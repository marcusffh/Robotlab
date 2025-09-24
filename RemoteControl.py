import curses ### 3rd library option. The others either wont work over linux, or not over a ssh connection. This one should work!!!!!
from Robotutils.CalibratedRobot import CalibratedRobot
import time

def RobotController(stdscr): ## stdscr is used to write text to the terminal
    robot = CalibratedRobot()
    speed = robot.default_speed # refers to calibrated Robot

    step_distance = 0.15 ## Meters
    step_angle = 10 # Degrees

    stdscr.nodelay(True)  # Makes sure the program doesnt freeze waiting for a key
    stdscr.clear()
    stdscr.addstr(0, 0, "Please use the arrow keys to control the robot")
    stdscr.addstr(1, 0, "Press q to terminate the program")
    stdscr.refresh()

    try:
        while True:
            key = stdscr.getch() ##gets the next key pressed

            if key == ord('q'): ## Terminate the program by typing q
                break

            # Continuous forward
            elif key == curses.KEY_UP:
                robot.drive(speed, speed, robot.FORWARD, robot.FORWARD)
                time.sleep(0.1)
                robot.stop()
                # Short press
                robot.drive_distance(step_distance, direction=robot.FORWARD, speed=speed)

            # Continuous backwards
            elif key == curses.KEY_DOWN:
                robot.drive(speed, speed, robot.BACKWARD, robot.BACKWARD)
                time.sleep(0.1)
                robot.stop()
                # Short press
                robot.drive_distance(step_distance, direction=robot.BACKWARD, speed=speed)

            # Continuous left
            elif key == curses.KEY_LEFT:
                robot.drive(speed, speed, robot.BACKWARD, robot.FORWARD)
                time.sleep(0.1)
                robot.stop()
                # Short press
                robot.turn_angle(step_angle, speed=speed)

            # Continuous right
            elif key == curses.KEY_RIGHT:
                robot.drive(speed, speed, robot.FORWARD, robot.BACKWARD)
                time.sleep(0.1)
                robot.stop()
                # Short press
                robot.turn_angle(-step_angle, speed=speed)

            else:
                robot.stop()

            time.sleep(0.1)

    except KeyboardInterrupt:
        stdscr.addstr(3, 0, "Interrupted by user")
    finally:
        robot.stop()

if __name__ == "__main__":
    curses.wrapper(RobotController)
