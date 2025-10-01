import cv2
import particle
import camera
import numpy as np
from timeit import default_timer as timer
import sys

# Flags
showGUI = True   # Whether or not to open GUI windows
onRobot = True   # Whether or not we are running on the Arlo robot

def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False."""
    return onRobot

if isRunningOnArlo():
    # XXX: Change this path to where your robot.py is located if needed
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

# Landmarks (IDs and known world positions in cm)
landmarkIDs = [9, 11]
landmarks = {
    9:  (0.0,   0.0),
    11: (300.0, 0.0)
}
landmark_colors = [CRED, CGREEN]

def jet(x):
    """Colour map for drawing particles based on weight."""
    r = (x >= 3.0/8.0 and x < 5.0/8.0) * (4.0 * x - 3.0/2.0) + (x >= 5.0/8.0 and x < 7.0/8.0) + (x >= 7.0/8.0) * (-4.0 * x + 9.0/2.0)
    g = (x >= 1.0/8.0 and x < 3.0/8.0) * (4.0 * x - 1.0/2.0) + (x >= 3.0/8.0 and x < 5.0/8.0) + (x >= 5.0/8.0 and x < 7.0/8.0) * (-4.0 * x + 7.0/2.0)
    b = (x < 1.0/8.0) * (4.0 * x + 1.0/2.0) + (x >= 1.0/8.0 and x < 3.0/8.0) + (x >= 3.0/8.0 and x < 5.0/8.0) * (-4.0 * x + 5.0/2.0)
    return (255.0*r, 255.0*g, 255.0*b)

def draw_world(est_pose, particles, world):
    """Draw particles, landmarks, and estimated pose in a simple world view."""
    # Screen transform (flip Y, add offsets to put origin on canvas)
    offsetX = 100
    offsetY = 250
    ymax = world.shape[0]
    world[:] = CWHITE

    # Largest weight (for color normalization)
    max_weight = 0.0
    for p in particles:
        max_weight = max(max_weight, p.getWeight())

    # Draw particles
    for p in particles:
        x = int(p.getX() + offsetX)
        y = ymax - (int(p.getY() + offsetY))
        colour = jet(p.getWeight()/max_weight) if max_weight > 0 else (128, 128, 128)
        cv2.circle(world, (x, y), 2, colour, 2)
        b = (int(p.getX() + 15.0*np.cos(p.getTheta()))+offsetX,
             ymax - (int(p.getY() + 15.0*np.sin(p.getTheta()))+offsetY))
        cv2.line(world, (x, y), b, colour, 2)

    # Draw landmarks
    for i, ID in enumerate(landmarkIDs):
        lm = (int(landmarks[ID][0] + offsetX), int(ymax - (landmarks[ID][1] + offsetY)))
        cv2.circle(world, lm, 5, landmark_colors[i], 2)

    # Draw estimated pose
    a = (int(est_pose.getX())+offsetX, ymax-(int(est_pose.getY())+offsetY))
    b = (int(est_pose.getX() + 15.0*np.cos(est_pose.getTheta()))+offsetX,
         ymax-(int(est_pose.getY() + 15.0*np.sin(est_pose.getTheta()))+offsetY))
    cv2.circle(world, a, 5, CMAGENTA, 2)
    cv2.line(world, a, b, CMAGENTA, 2)

def initialize_particles(num_particles):
    particles = []
    for _ in range(num_particles):
        # Uniform-ish prior in a rectangle around the landmark line
        p = particle.Particle(
            600.0*np.random.ranf() - 100.0,           # x in [-100, 500]
            600.0*np.random.ranf() - 250.0,           # y in [-250, 350]
            np.mod(2.0*np.pi*np.random.ranf(), 2.0*np.pi),
            1.0/num_particles
        )
        particles.append(p)
    return particles

# -------------------- Main program --------------------
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
    est_pose = particle.estimate_pose(particles)

    # >>> ADDED — Navigation state (SEARCH -> DRIVE_TO_CENTER -> ARRIVED)
    nav_state = "SEARCH"
    target_xy = (150.0, 0.0)   # midpoint between ID 9 (0,0) and ID 11 (300,0)
    # Control gains / limits
    K_TH = 1.2                  # [rad/s per rad] heading P-gain
    K_V  = 0.6                  # [cm/s per cm]   speed P-gain
    MAX_W = 1.0                 # rad/s clamp
    MAX_V = 20.0                # cm/s clamp
    ARRIVE_DIST = 10.0          # cm
    ARRIVE_TH   = np.deg2rad(9) # rad
    # <<< ADDED

    # Driving (cmd) variables
    cmd_v = 0.0          # cm/s
    cmd_w = 0.0          # rad/s

    # Robot init
    arlo = None
    if isRunningOnArlo():
        try:
            import robot
            arlo = robot.Robot()
            print("Arlo robot initialized.")
        except Exception as e:
            print("Warning: could not initialize Arlo robot:", e)
            arlo = None

    # Timer for integrating motion
    _last_time = timer()

    # World canvas
    world = np.zeros((500, 500, 3), dtype=np.uint8)
    draw_world(est_pose, particles, world)

    # Camera
    print("Opening and initializing camera")
    if isRunningOnArlo():
        cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)
    else:
        cam = camera.Camera(1, robottype='macbookpro', useCaptureThread=False)

    # --- Main loop ---
    while True:
        # Handle quit key (GUI only)
        if showGUI:
            action = cv2.waitKey(1)
            if action == ord('q'):
                break

        # ======= 1) Get frame & detect ArUco landmarks =======
        colour = cam.get_next_frame()                  # BGR frame
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)

        # Build per-ID best (closest) observation this frame
        observations = {}
        if objectIDs is not None:
            for i in range(len(objectIDs)):
                ID = int(objectIDs[i])
                if ID in landmarkIDs:
                    d = float(dists[i])    # [cm]
                    a = float(angles[i])   # [rad], positive left
                    # Keep the closest for each ID
                    if (ID not in observations) or (d < observations[ID][0]):
                        observations[ID] = (d, a)

        # ======= 2) Measurement update (weights) =======
        if len(observations) > 0:
            # >>> ADDED — Gaussian distance & bearing likelihoods
            sigma_d  = 8.0                 # [cm]   tune
            sigma_th = np.deg2rad(5.0)     # [rad]  tune
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
                    w *= inv_norm_d * np.exp(-0.5 * ((d_meas - d_pred)/sigma_d)**2) \
                       * inv_norm_th * np.exp(-0.5 * (a_err/sigma_th)**2)
                p.setWeight(w)

            # Normalize weights
            S = sum(p.getWeight() for p in particles)
            if S > 0.0:
                for p in particles:
                    p.setWeight(p.getWeight()/S)
            else:
                # fallback to uniform if all zero
                for p in particles:
                    p.setWeight(1.0/num_particles)
            # <<< ADDED
        else:
            # No observation — keep weights uniform (don’t over-trust motion alone)
            for p in particles:
                p.setWeight(1.0/num_particles)

        # ======= 3) Resampling (systematic / low-variance) =======
        # (No helper in particle.py for this; implement here)
        N = len(particles)
        weights = np.array([p.getWeight() for p in particles], dtype=float)
        wsum = weights.sum()
        if wsum == 0.0:
            weights[:] = 1.0/N
        else:
            weights /= wsum

        r = np.random.rand() / N
        c = weights[0]
        i = 0
        new_particles = []
        for m in range(N):
            U = r + m * (1.0 / N)
            while U > c and i < N-1:
                i += 1
                c += weights[i]
            pi_sel = particles[i]
            new_particles.append(particle.Particle(pi_sel.getX(), pi_sel.getY(), pi_sel.getTheta(), 1.0/N))
        particles = new_particles

        # ======= 4) Pose estimate from particles =======
        est_pose = particle.estimate_pose(particles)

        # ======= 5) Navigation state machine & control =======
        # >>> ADDED — Rotate to see both IDs, then drive to center (150, 0)
        saw_both = all(ID in observations for ID in landmarkIDs)

        if nav_state == "SEARCH":
            if saw_both:
                nav_state = "DRIVE_TO_CENTER"
                cmd_v, cmd_w = 0.0, 0.0
            else:
                # rotate in place slowly (left)
                cmd_v = 0.0
                cmd_w = +0.6   # rad/s

        elif nav_state == "DRIVE_TO_CENTER":
            # Go-to-goal controller in world frame using estimated pose
            tx, ty = target_xy
            dx = tx - est_pose.getX()
            dy = ty - est_pose.getY()
            dist = np.hypot(dx, dy)
            # bearing error relative to robot heading
            desired_th = np.arctan2(dy, dx)
            e_th = (desired_th - est_pose.getTheta() + np.pi) % (2*np.pi) - np.pi

            # P-controllers with simple gating: slow down if heading error large
            cmd_w = np.clip(K_TH * e_th, -MAX_W, MAX_W)
            speed_gain = 0.3 if abs(e_th) > np.deg2rad(20) else 1.0
            cmd_v = np.clip(K_V * dist * speed_gain, -MAX_V, MAX_V)

            if (dist < ARRIVE_DIST) and (abs(e_th) < ARRIVE_TH):
                nav_state = "ARRIVED"
                cmd_v, cmd_w = 0.0, 0.0

        elif nav_state == "ARRIVED":
            cmd_v, cmd_w = 0.0, 0.0
        # <<< ADDED

        # ======= 6) Send motion to robot (if present) =======
        # Map (cmd_v [cm/s], cmd_w [rad/s]) to wheel commands; adjust gains for your robot
        if arlo is not None:
            try:
                base  = int(np.clip(cmd_v * 2.0,  -127, 127))      # cm/s -> PWM-ish
                steer = int(np.clip(cmd_w * 50.0, -127, 127))      # rad/s -> PWM-ish
                left  = int(np.clip(base - steer, -127, 127))
                right = int(np.clip(base + steer, -127, 127))
                # Flags: 1 means forward enabled for each motor (library-specific)
                arlo.go_diff(left, right, 1)
            except Exception as e:
                print("Drive warning:", e)

        # ======= 7) Motion prediction on particles using particle.py helpers =======
        now = timer()
        dt = max(1e-3, now - _last_time)
        _last_time = now

        # Local-frame deltas: forward = v*dt, lateral ~ 0, heading change = w*dt
        d_x     = cmd_v * dt        # cm
        d_y     = 0.0               # cm (no commanded lateral motion)
        d_theta = cmd_w * dt        # rad

        # Apply commanded motion to each particle (local -> world handled by move_particle)
        for p in particles:
            particle.move_particle(p, d_x, d_y, d_theta)

        # Add small process noise to keep diversity (using particle.py)
        particle.add_uncertainty(particles, sigma=0.5, sigma_theta=np.deg2rad(1.5))

        # ======= 8) Draw & show =======
        if showGUI:
            draw_world(est_pose, particles, world)
            cam.draw_aruco_objects(colour)  # overlays in-place

            cv2.imshow(WIN_RF1, colour)
            cv2.imshow(WIN_World, world)

finally:
    # Clean up
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        cam.terminateCaptureThread()
    except Exception:
        pass
