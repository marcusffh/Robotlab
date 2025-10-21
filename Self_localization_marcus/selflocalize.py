import cv2
import particle
import camera
import numpy as np
import time
from timeit import default_timer as timer
import sys
from Robotutils.CalibratedRobot import CalibratedRobot 


# Flags
showGUI = False  # Whether or not to open GUI windows
onRobot = True  # Whether or not we are running on the Arlo robot


def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
      You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot


if isRunningOnArlo():
    try:
        from Robotutils.robot import robot
    except ImportError:
        print("robot_utils module not found")
        onRobot = False
else:
    onRobot = False




# Some color constants in BGR format
CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CCYAN = (255, 255, 0)
CYELLOW = (0, 255, 255)
CMAGENTA = (255, 0, 255)
CWHITE = (255, 255, 255)
CBLACK = (0, 0, 0)

# Landmarks.
# The robot knows the position of 2 landmarks. Their coordinates are in the unit centimeters [cm].
landmarkIDs = [6, 7]
landmarks = {
    6: (0.0, 0.0),  # Coordinates for landmark 1
    7: (300.0, 0.0)  # Coordinates for landmark 2
}
landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks

# Convert landmarks to numpy arrays for compatibility with particle filter
LANDMARKS = {
    6: np.array([0.0, 0.0]),
    7: np.array([300.0, 0.0])
}



def jet(x):
    """Colour map for drawing particles. This function determines the colour of 
    a particle from its weight."""
    r = (x >= 3.0/8.0 and x < 5.0/8.0) * (4.0 * x - 3.0/2.0) + (x >= 5.0/8.0 and x < 7.0/8.0) + (x >= 7.0/8.0) * (-4.0 * x + 9.0/2.0)
    g = (x >= 1.0/8.0 and x < 3.0/8.0) * (4.0 * x - 1.0/2.0) + (x >= 3.0/8.0 and x < 5.0/8.0) + (x >= 5.0/8.0 and x < 7.0/8.0) * (-4.0 * x + 7.0/2.0)
    b = (x < 1.0/8.0) * (4.0 * x + 1.0/2.0) + (x >= 1.0/8.0 and x < 3.0/8.0) + (x >= 3.0/8.0 and x < 5.0/8.0) * (-4.0 * x + 5.0/2.0)

    return (255.0*r, 255.0*g, 255.0*b)

def draw_world(est_pose, particles, world):
    """Visualization.
    This functions draws robots position in the world coordinate system."""

    # Fix the origin of the coordinate system
    offsetX = 100
    offsetY = 250

    # Constant needed for transforming from world coordinates to screen coordinates (flip the y-axis)
    ymax = world.shape[0]

    world[:] = CWHITE # Clear background to white

    # Find largest weight
    max_weight = 0
    for particle in particles:
        max_weight = max(max_weight, particle.getWeight())

    # Draw particles
    for particle in particles:
        x = int(particle.getX() + offsetX)
        y = ymax - (int(particle.getY() + offsetY))
        colour = jet(particle.getWeight() / max_weight)
        cv2.circle(world, (x,y), 2, colour, 2)
        b = (int(particle.getX() + 15.0*np.cos(particle.getTheta()))+offsetX, 
                                     ymax - (int(particle.getY() + 15.0*np.sin(particle.getTheta()))+offsetY))
        cv2.line(world, (x,y), b, colour, 2)

    # Draw landmarks
    for i in range(len(landmarkIDs)):
        ID = landmarkIDs[i]
        lm = (int(landmarks[ID][0] + offsetX), int(ymax - (landmarks[ID][1] + offsetY)))
        cv2.circle(world, lm, 5, landmark_colors[i], 2)

    # Draw estimated robot pose
    a = (int(est_pose.getX())+offsetX, ymax-(int(est_pose.getY())+offsetY))
    b = (int(est_pose.getX() + 15.0*np.cos(est_pose.getTheta()))+offsetX, 
         ymax-(int(est_pose.getY() + 15.0*np.sin(est_pose.getTheta()))+offsetY))
    cv2.circle(world, a, 5, CMAGENTA, 2)
    cv2.line(world, a, b, CMAGENTA, 2)




# Main program #
try:
    if showGUI:
        # Open windows
        WIN_RF1 = "Robot view"
        cv2.namedWindow(WIN_RF1)
        cv2.moveWindow(WIN_RF1, 50, 50)

        WIN_World = "World view"
        cv2.namedWindow(WIN_World)
        cv2.moveWindow(WIN_World, 500, 50)

    # Initialize particles
    num_particles = 1000
    particles = particle.initialize_particles(num_particles)
    est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

    # Driving parameters
    velocity = 0.0 # cm/sec
    angular_velocity = 0.0 # radians/sec

    # Initialize the robot 
    if isRunningOnArlo():
        r = robot.Robot()
        cal_robot = CalibratedRobot()
    else:
        r = None

    # Allocate space for world map
    world = np.zeros((500,500,3), dtype=np.uint8)

    # Draw map
    draw_world(est_pose, particles, world)

    print("Opening and initializing camera")
    if isRunningOnArlo():
        #cam = camera.Camera(0, robottype='arlo', useCaptureThread=True)
        cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)
    else:
        #cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=True)
        cam = camera.Camera(1, robottype='macbookpro', useCaptureThread=False)

    objectIDs = None
    while True:

        # Move the robot according to user input (only for testing)
        action = cv2.waitKey(10)
        if action == ord('q'): # Quit
            break
    
        if not isRunningOnArlo():
            if action == ord('w'): # Forward
                velocity += 4.0
            elif action == ord('x'): # Backwards
                velocity -= 4.0
            elif action == ord('s'): # Stop
                velocity = 0.0
                angular_velocity = 0.0
            elif action == ord('a'): # Left
                angular_velocity += 0.2
            elif action == ord('d'): # Right
                angular_velocity -= 0.2


        # Use motor controls to update particles
        if isRunningOnArlo():
            #The case where we havent seen both landmarks
            if objectIDs is None or len(set(objectIDs)) < 2:
                print("Searching landmarks")
                cal_robot.turn_angle(10)

                #update particle based on turn
                distance_cm, angle_rad = particle.robot_command_to_motion(turn_degrees=10, drive_meters= 0.0)
                particle.prediction_step(particles, distance_cm, angle_rad)
            #Bot landmarks detected
            else: 
                print("both landmarks detected, moving to midpoint")

                # Compute target (midpoint between landmarks)
                lm1, lm2 = LANDMARKS[6], LANDMARKS[7]
                target_x = (lm1[0] + lm2[0]) / 2.0
                target_y = (lm1[1] + lm2[1]) / 2.0

                #estimate current pose
                est_pose = particle.estimate_pose(particles)
                dx = target_x - est_pose.getX()
                dy = target_y - est_pose.getY()

                #compute distance and heading to midpoint
                distance_cm = np.sqrt(dx**2 + dy**2)
                desired_angle = np.arctan2(dy, dx)
                angle_diff = desired_angle - est_pose.getTheta()
                angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                
                #turn toward the 
                turn_degrees = np.rad2deg(angle_diff)
                cal_robot.turn_angle(turn_degrees)

                #update particle for turn
                d_cm, angle_rad = particle.robot_command_to_motion(turn_degrees= turn_degrees , drive_meters = 0.0)
                particle.prediction_step(particles, 0.0, angle_rad)

                #Drive to midpoint
                drive_meters = distance_cm /100
                cal_robot.drive(drive_meters)

                #update particles for drive
                d_cm, angle_rad = particle.robot_command_to_motion(turn_degrees= 0.0 , drive_meters= drive_meters)
                particle.prediction_step(particles, d_cm, angle_rad)

                print("reached midpoint. terminating")
                cal_robot.stop()
                break

          
        # ===== END OF ROBOT MOVEMENT CODE =====

      
        # Fetch next frame
        colour = cam.get_next_frame()
        
        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if not isinstance(objectIDs, type(None)):
            # List detected objects
            for i in range(len(objectIDs)):
                print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])
                # XXX: Do something for each detected object - remember, the same ID may appear several times

            # Compute particle weights (correction step)
            particle.correction_step(particles, objectIDs, dists, angles, LANDMARKS,
                                    sigma_d=10.0, sigma_theta=np.deg2rad(10))

            # Resampling step
            particles = particle.resampling_step(particles)

            # Draw detected objects
            cam.draw_aruco_objects(colour)
        else:
            # No observation - reset weights to uniform distribution
            for p in particles:
                p.setWeight(1.0/num_particles)

    
        est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

        if showGUI:
            # Draw map
            draw_world(est_pose, particles, world)
    
            # Show frame
            cv2.imshow(WIN_RF1, colour)

            # Show world
            cv2.imshow(WIN_World, world)
    
  
finally: 
    # Make sure to clean up even if an exception occurred
    
    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    cam.terminateCaptureThread()