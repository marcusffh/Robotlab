import numpy as np
import random_numbers as rn

"""
Please remember that each particle represents on hypothesis of where the robot could be

"""



#Given
class Particle(object):
    """Data structure for storing particle information (state and weight)"""
    def __init__(self, x=0.0, y=0.0, theta=0.0, weight=0.0):
        self.x = x #x coordinate
        self.y = y #y coordinate
        self.theta = np.mod(theta, 2.0*np.pi) # wraps angle into raidans  [0 , 2\pi]
        self.weight = weight ## How important is this particle

    def getX(self):
        return self.x
        
    def getY(self):
        return self.y
        
    def getTheta(self):
        return self.theta
        
    def getWeight(self):
        return self.weight

    def setX(self, val):
        self.x = val

    def setY(self, val):
        self.y = val

    def setTheta(self, val):
        self.theta = np.mod(val, 2.0*np.pi)

    def setWeight(self, val):
        self.weight = val


#Given
def estimate_pose(particles_list):
    """Estimate the pose from particles by computing the average position and orientation over all particles. 
    This is not done using the particle weights, but just the sample distribution."""
    x_sum = 0.0
    y_sum = 0.0
    cos_sum = 0.0
    sin_sum = 0.0
     
    for particle in particles_list:
        x_sum += particle.getX()
        y_sum += particle.getY()
        cos_sum += np.cos(particle.getTheta())
        sin_sum += np.sin(particle.getTheta())
        
    flen = len(particles_list)
    if flen != 0:
        x = x_sum / flen
        y = y_sum / flen
        theta = np.arctan2(sin_sum/flen, cos_sum/flen)
    else:
        x = x_sum
        y = y_sum
        theta = 0.0
        
    return Particle(x, y, theta)
     

## This is the only function that we were to impliment
def move_particle(particle, delta_x, delta_y, delta_theta):
    """Move the particle by (delta_x, delta_y, delta_theta)
    Warning: we are assuming that delta_x and delta_y are given
    in world coordinates, this will not work if they are given in robot coordinates.
    """
    particle.x += delta_x
    particle.y += delta_y
    particle.theta = np.mod(particle.theta + delta_theta, 2.0 * np.pi)


#Given
def add_uncertainty(particles_list, sigma, sigma_theta):
    """Add some noise to each particle in the list. Sigma and sigma_theta is the noise
    variances for position and angle noise."""
    for particle in particles_list:
        particle.x += rn.randn(0.0, sigma)
        particle.y += rn.randn(0.0, sigma)
        particle.theta = np.mod(particle.theta + rn.randn(0.0, sigma_theta), 2.0 * np.pi) 


#Given, but we probaly wont use it...
def add_uncertainty_von_mises(particles_list, sigma, theta_kappa):
    """Add some noise to each particle in the list. Sigma and theta_kappa is the noise
    variances for position and angle noise."""
    for particle in particles_list:
        particle.x += rn.randn(0.0, sigma)
        particle.y += rn.randn(0.0, sigma)
        particle.theta = np.mod(rn.rand_von_mises(particle.theta, theta_kappa), 2.0 * np.pi) - np.pi

#impiment motion model
#Used for the prediction step


# ------------------------ Self implimented particle filter---------------------------------------------

# PREDICTION STEP (SAMPLING STEP)  
# The motion modeL
def motion_model_odometry(particle, distance, angle_change, sigma_d=2.0, sigma_theta=0.05):
    """
    A motion model is a mathematical model that describes how the robots pose is likely
    to change from one time step to the next, given its control inputs, It is used in the 
    prediction step of the particle filter. It helps predict where the robot is after a
    moveset.
    
    Args:
        particle: Particle object to update
        distance: Distance driven in cm
        angle_change: Rotation angle in radians
        sigma_d: Standard deviation for distance noise (cm). Please Tune as neede
        sigma_theta: Standard deviation for angular noise (radians). Please tune as needed
    """
    
    # Step 1: Turn with noise
    noisy_turn = angle_change + rn.randn(0.0, sigma_theta)
    particle.theta = np.mod(particle.theta + noisy_turn, 2.0 * np.pi)
    
    # Step 2: Drive with noise in the new orientation
    noisy_distance = distance + rn.randn(0.0, sigma_d)
    particle.x += noisy_distance * np.cos(particle.theta)
    particle.y += noisy_distance * np.sin(particle.theta)

#Converter
def robot_command_to_motion(turn_degrees, drive_meters):
    """
    Converts the calibratedRobot turn and drive functions, into input that the motion-
    model can take.
    
    Args:
        turn_degrees: Angle turned by robot (positive = left, negative = right)
        drive_meters: Distance driven in meters
    
    Returns:
        distance (cm), angle_change (radians)
    """
    distance_cm = drive_meters * 100.0  # Convert meters to cm
    angle_rad = np.deg2rad(turn_degrees)  # Convert degrees to radians
    
    return distance_cm, angle_rad

# actual prediction step
def prediction_step(particles, distance, angle_change, sigma_d=2.0, sigma_theta=0.05):
    """
    Apply motion model to ALL particles.
    
    Args:
        particles: List of Particle objects
        distance: Distance the robot drove in cm
        angle_change: Angle the robot turned in radians
        sigma_d: Position uncertainty (cm)
        sigma_theta: Angular uncertainty (radians)
    """
    for particle in particles:
        motion_model_odometry(particle, distance, angle_change, sigma_d, sigma_theta)

#----------------------------------------------------------------------------------------------------

# CORRECTION STEP (WEIGHTING STEP)
def correction_step(particles, ids, dists, angles,LANDMARKS, sigma_d=10.0, sigma_theta=np.deg2rad(10)):
    """
    Update particle weights based on landmark observations. The code is based on algorithm 3 from
    https://medium.com/@mathiasmantelli/particle-filter-part-4-pseudocode-and-python-code-052a74236ba4
    It also needs to know which landmarks it should look for create something like
    
    Args:
        particles: List of Particle objects
        ids: ArUco IDs from camera (numpy array or None)
        dists: Distances in cm from camera (numpy array or None)
        angles: Angles in radians from camera (numpy array or None)
        LANDMARKS: dict mapping landmark id to coordinates, see example below
            LANDMARKS = {
                1: np.array([0.0, 0.0]),
                2: np.array([300.0, 0.0])}
        sigma_d: Distance measurement noise (cm)
        sigma_theta: Angle measurement noise (radians)
    """
    # No observations? Keep uniform weights
    if ids is None:
        for p in particles:
            p.weight = 1.0 / len(particles)
        return
    
    # Outer loop (Particle)
    # For each particle, compute weight
    for particle in particles:
        weight = 1.0
        
        #Inner Loop (Landmarks)
        # For each observed landmark
        for i in range(len(ids)):
            landmark_id = int(ids[i])
            
            if landmark_id not in LANDMARKS:
                continue  # Skip unknown landmarks
            
            landmark_pos = LANDMARKS[landmark_id]
            

            # -------------------------------
            # Distance Part (Formula 2)
            # -------------------------------
            #d^{(i)} = \sqrt{(l_x -x^{(i)})^2 + (l_y - y^{(i)})^2} essentially
            dx = landmark_pos[0] - particle.x #(l_x - x^(i))
            dy = landmark_pos[1] - particle.y #(l_y - y^(i))
            euclidean_distance = np.sqrt(dx**2 + dy**2) #euclidean distance

            #coefficient for Gaussian distance
            coef_distance = 1 / (np.sqrt(2 * np.pi * sigma_d**2))

            # Equation 2 from the pdf: distance probability
            prob_dist = coef_distance * np.exp(- (dists[i] - euclidean_distance)**2 / ( 2 * sigma_d**2)) 



            
            # -------------------------------
            # Orientation Part (Formula 3)
            # -------------------------------

            #phi(i)
            angle_world = np.arctan2(dy, dx)  # angle from particle to landmark in world frame
            phi_i = angle_world - particle.theta  # expected orientation measurement, relative to robot's current position
            phi_i = np.arctan2(np.sin(phi_i), np.cos(phi_i))  # wrap angle to [-pi, pi]

            #phi_M
            phi_M = angles[i]  # orientation measurement, relative to robot's current position

            # difference between angles (wrapped [-pi , pi])
            angle_diff = phi_M - phi_i
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff)) #wrap angle

            #coefficient for Gaussian distance
            coef_orientation = 1 / (np.sqrt(2 * np.pi * sigma_theta**2))
        
            #Prob dist using equation 3 from the pdf
            prob_orientation = coef_orientation * np.exp(- (angle_diff)**2 / ( 2 * sigma_theta**2))

            #calculate weight for this landmark observation
            weight *= prob_dist * prob_orientation

        
        particle.weight = weight
    
    # Normalize weights
    total = sum(p.weight for p in particles)
    if total > 0:
        for p in particles:
            p.weight /= total
    else:
        # All weights zero? Reset to uniform
        for p in particles:
            p.weight = 1.0 / len(particles)

#------------------------------------------------------
#The Resampling step
def resampling_step(particles):

    N = len(particles) # How many particles are we resampling
    if N == 0:     
        return []
    
    #Get weights
    w = np.array([p.weight for p in particles])  # FIXED: Added brackets
    w = w / np.sum(w)

    #Compute the cumulative distribution
    H = np.cumsum(w)

    # --- Resampling ---
    resampled_particles = []
    for _ in range(N):
        z = np.random.ranf()  # FIXED: Changed from rn.rand_uniform(0.0, 1.0) to np.random.ranf()
        i = np.searchsorted(H, z)
        selected = particles[i]
        # Copy selected particle into new set
        new_p = Particle(selected.x, selected.y, selected.theta, 1.0 / N)
        resampled_particles.append(new_p)

    return resampled_particles