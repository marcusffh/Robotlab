import cv2
import particle
import camera
import numpy as np
import time
from timeit import default_timer as timer
import sys
#from Robotutils.CalibratedRobot import CalibratedRobot 


# Flags
showGUI = False  # Whether or not to open GUI windows
onRobot = True  # Whether or not we are running on the Arlo robot


def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
      You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot


if isRunningOnArlo():
    # XXX: You need to change this path to point to where your robot.py file is located
    sys.path.append("Robotlab/Robotutils/")


try:
    import robot
    onRobot = True
except ImportError:
    print("selflocalize.py: robot module not present - forcing not running on Arlo!")
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
landmarkIDs = [1, 2]
landmarks = {
    1: (0.0, 0.0),  # Coordinates for landmark 1
    2: (300.0, 0.0)  # Coordinates for landmark 2
}
landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks





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



def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        # Random starting points. 
        p = particle.Particle(600.0*np.random.ranf() - 100.0, 600.0*np.random.ranf() - 250.0, np.mod(2.0*np.pi*np.random.ranf(), 2.0*np.pi), 1.0/num_particles)
        particles.append(p)

    return particles


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
    particles = initialize_particles(num_particles)

    est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

    # Driving parameters
    velocity = 0.0 # cm/sec
    angular_velocity = 0.0 # radians/sec

    # Initialize the robot (XXX: You do this)
    arlo = None
    if isRunningOnArlo():
        try:
            import robot
            arlo = robot.Robot()
            print("Arlo robot initialized.")
        except Exception as e:
            print("Warning: could not initialize Arlo robot:", e)
            arlo = None

# Timer used for odometry integration of the motion model
    _last_time = timer()

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
        # XXX: Make the robot drive
        # XXX: You do this
        
    
        # >>> ADDED — send motion to Arlo (if present) and propagate particles with motion model
# 2a) compute time step
        now = timer()
        dt = max(1e-3, now - _last_time)
        _last_time = now

# 2b) drive the robot (safe clamped mapping to go_diff). Adjust gains if needed on your robot.
        if arlo is not None:
            try:
                base  = int(np.clip(velocity * 2.0,  -127, 127))   # rough cm/s -> pwm scaling
                steer = int(np.clip(angular_velocity * 50.0, -127, 127))  # rad/s -> pwm scaling
                left  = int(np.clip(base - steer, -127, 127))
                right = int(np.clip(base + steer, -127, 127))
                arlo.go_diff(left, right, 1)  # 1 = forward flag on both wheels
            except Exception as e:
        # If the API differs on your Arlo lib, this prevents a crash during development
                print("Drive warning:", e)

# 2c) propagate particles (rotate–translate–rotate + small noise)
        new_particles = []
        for p in particles:
            th = p.getTheta()

    # split rotation to reduce bias; add small Gaussian noise (tune if needed)
            drot1 = (angular_velocity * dt) * 0.5 + np.random.normal(0.0, 0.01)
            dtran = (velocity * dt)          + np.random.normal(0.0, 0.5)   # cm noise
            drot2 = (angular_velocity * dt) * 0.5 + np.random.normal(0.0, 0.01)

            th1 = th + drot1
            x   = p.getX() + dtran * np.cos(th1)
            y   = p.getY() + dtran * np.sin(th1)
            th2 = th1 + drot2

    # wrap to [-pi, pi]
            th2 = (th2 + np.pi) % (2*np.pi) - np.pi

            new_particles.append(particle.Particle(x, y, th2, p.getWeight()))
        particles = new_particles
# <<< ADDED



        # Fetch next frame
        colour = cam.get_next_frame()
        
        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if not isinstance(objectIDs, type(None)):
            # List detected objects
            for i in range(len(objectIDs)):
                print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])
                # XXX: Do something for each detected object - remember, the same ID may appear several times
                # >>> ADDED — keep the closest observation per landmark ID this frame
                if 'observations' not in locals():
                    observations = {}
                ID = int(objectIDs[i])
                if ID in landmarkIDs:
                    d = float(dists[i])
                    a = float(angles[i])  # radians, bearing in robot frame
                    if (ID not in observations) or (d < observations[ID][0]):
                        observations[ID] = (d, a)
# <<< ADDED

            # Compute particle weights
            # XXX: You do this
            if 'observations' in locals() and len(observations) > 0:
                sigma_d  = 8.0                   # distance std [cm] — tune to your camera
                sigma_th = np.deg2rad(5.0)       # bearing std [rad] — tune to your camera

                inv_norm_d  = 1.0 / (np.sqrt(2*np.pi) * sigma_d)
                inv_norm_th = 1.0 / (np.sqrt(2*np.pi) * sigma_th)

                for p in particles:
                    x, y, th = p.getX(), p.getY(), p.getTheta()
                    w = 1.0
                    for ID, (d_meas, a_meas) in observations.items():
                        lx, ly = landmarks[ID]
                        dx, dy = lx - x, ly - y
                        d_pred = np.hypot(dx, dy)
                        a_pred = np.arctan2(dy, dx) - th
            # wrap angle residual to [-pi, pi]
                        a_err = ( (a_meas - a_pred + np.pi) % (2*np.pi) ) - np.pi
            # Gaussian(d)*Gaussian(angle)
                        w *= inv_norm_d  * np.exp(-0.5 * ((d_meas - d_pred)/sigma_d)**2) \
                        * inv_norm_th * np.exp(-0.5 * (a_err/sigma_th)**2)
                    p.setWeight(w)

    # normalize weights
                S = sum(p.getWeight() for p in particles)
                if S > 0.0:
                    for p in particles:
                        p.setWeight(p.getWeight()/S)

    # clear per-frame observations
                observations = {}
# <<< ADDED

            # Resampling
            # XXX: You do this
            # >>> ADDED — low-variance (systematic) resampling
            N = len(particles)
            weights = np.array([p.getWeight() for p in particles], dtype=float)
            wsum = weights.sum()
            if wsum == 0.0:
                weights = np.ones(N, dtype=float) / N
            else:
                weights /= wsum

# systematic resampling
            r = np.random.rand() / N
            c = weights[0]
            i = 0
            new_particles = []
            for m in range(N):
                U = r + m * (1.0 / N)
                while U > c and i < N-1:
                    i += 1
                    c += weights[i]
                pi = particles[i]
                new_particles.append(particle.Particle(pi.getX(), pi.getY(), pi.getTheta(), 1.0 / N))
            particles = new_particles
# <<< ADDED


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

