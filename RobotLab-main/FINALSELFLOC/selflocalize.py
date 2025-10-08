import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import particle
import camera
import numpy as np
import time
from timeit import default_timer as timer
from RobotUtils.CalibratedRobot import CalibratedRobot
from scipy.stats import norm
import math
from LocalizationPathing import LocalizationPathing
import random
import cv2
from LandmarkOccupancyGrid import LandmarkOccupancyGrid

# Flags
showGUI = False  # GUI optional
onRobot = True   # running on Arlo

def isRunningOnArlo():
    return onRobot

try:
    from RobotUtils.Robot import Robot
except ImportError:
    print("selflocalize.py: robot module not present - forcing not running on Arlo!")
    onRobot = False

# Colors
CRED=(0,0,255); CGREEN=(0,255,0); CMAGENTA=(255,0,255); CWHITE=(255,255,255)

# Landmarks (cm)
landmarkIDs = [6, 7]
landmarks = {6:(0.0,0.0), 7:(300.0,0.0)}
center = np.array([(landmarks[6][0]+landmarks[7][0])/2,
                   (landmarks[6][1]+landmarks[7][1])/2])  # -> (150, 0)

def jet(x):
    r=(x>=3/8 and x<5/8)*(4*x-1.5)+(x>=5/8 and x<7/8)+(x>=7/8)*(-4*x+4.5)
    g=(x>=1/8 and x<3/8)*(4*x-0.5)+(x>=3/8 and x<5/8)+(x>=5/8 and x<7/8)*(-4*x+3.5)
    b=(x<1/8)*(4*x+0.5)+(x>=1/8 and x<3/8)+(x>=3/8 and x<5/8)*(-4*x+2.5)
    return (255.0*r,255.0*g,255.0*b)

def draw_world(est_pose, particles, world):
    offsetX=100; offsetY=250; ymax=world.shape[0]; world[:]=CWHITE
    mw=max([p.getWeight() for p in particles]+[0.0])
    for p in particles:
        x=int(p.getX()+offsetX); y=ymax-(int(p.getY()+offsetY))
        colour=jet((p.getWeight()/mw) if mw>0 else 0.0)
        cv2.circle(world,(x,y),2,colour,2)
        b=(int(p.getX()+15.0*np.cos(p.getTheta()))+offsetX,
           ymax-(int(p.getY()+15.0*np.sin(p.getTheta()))+offsetY))
        cv2.line(world,(x,y),b,colour,2)
    for ID,col in zip(landmarkIDs,[CRED, CGREEN]):
        lm=(int(landmarks[ID][0]+offsetX), int(ymax-(landmarks[ID][1]+offsetY)))
        cv2.circle(world,lm,5,col,2)
    a=(int(est_pose.getX())+offsetX, ymax-(int(est_pose.getY()+offsetY)))
    b=(int(est_pose.getX()+15.0*np.cos(est_pose.getTheta()))+offsetX,
       ymax-(int(est_pose.getY()+15.0*np.sin(est_pose.getTheta()))+offsetY))
    cv2.circle(world,a,5,CMAGENTA,2); cv2.line(world,a,b,CMAGENTA,2)

def initialize_particles(N):
    return [particle.Particle(600.0*np.random.ranf()-100.0,
                              600.0*np.random.ranf()-250.0,
                              np.mod(2.0*np.pi*np.random.ranf(),2.0*np.pi),
                              1.0/N) for _ in range(N)]

def sample_motion_model(particles_list, distance, angle, sigma_d, sigma_theta):
    for p in particles_list:
        dx=distance*np.cos(p.getTheta()+angle); dy=distance*np.sin(p.getTheta()+angle)
        particle.move_particle(p, dx, dy, angle)
    particle.add_uncertainty_von_mises(particles_list, sigma_d, sigma_theta)

def measurement_model(particle_list, landmarkIDs, dists, angles, sigma_d, sigma_theta):
    for p in particle_list:
        x_i=p.getX(); y_i=p.getY(); th=p.getTheta(); p_obs=1.0
        for ID,dist,ang in zip(landmarkIDs,dists,angles):
            if ID in landmarkIDs:
                lx,ly=landmarks[ID]; d_i=np.hypot(lx-x_i, ly-y_i); d_i=max(d_i,1e-9)
                p_d=norm.pdf(dist,loc=d_i,scale=sigma_d)
                e_th=np.array([np.cos(th),np.sin(th)]); e_th_hat=np.array([-np.sin(th),np.cos(th)])
                e_l=np.array([lx-x_i,ly-y_i])/d_i
                dot=float(np.clip(np.dot(e_l,e_th),-1.0,1.0))
                phi=np.sign(np.dot(e_l,e_th_hat))*np.arccos(dot)
                p_phi=norm.pdf(ang,loc=phi,scale=sigma_theta)
                p_obs*=p_d*p_phi
        p.setWeight(p_obs)

def resample_particles(particle_list, weights, w_fast, w_slow):
    cdf=np.cumsum(weights); res=[]
    for _ in range(len(particle_list)):
        ratio=(w_fast/w_slow) if w_slow>0 else 1.0
        if random.random()<max(0.0,1.0-ratio):
            res.append(initialize_particles(1)[0])
        else:
            z=np.random.rand(); idx=np.searchsorted(cdf,z); src=particle_list[idx]
            res.append(particle.Particle(src.getX(),src.getY(),src.getTheta(),1.0/len(particle_list)))
    return res

def filter_landmarks_by_distance(objectIDs,dists,angles):
    md={}
    for id_,d,a in zip(objectIDs,dists,angles):
        if id_ not in md or d<md[id_][0]: md[id_]=(d,a)
    f_ids=list(md.keys()); f_d=[md[ID][0] for ID in f_ids]; f_a=[md[ID][1] for ID in f_ids]
    return f_ids,f_d,f_a

# ---- robust 'are we at the midpoint?' check using PF pose ----
CENTER_TOL_CM = 10.0
GOAL_HITS_REQUIRED = 2

try:
    if showGUI:
        cv2.namedWindow("Robot view"); cv2.moveWindow("Robot view",50,50)
        cv2.namedWindow("World view"); cv2.moveWindow("World view",500,50)

    num_particles=1000
    particles=initialize_particles(num_particles)
    est_pose=particle.estimate_pose(particles)
    print(f"estimated pose: {est_pose}")

    distance=0.0
    angle=0.0

    sigma_d=9.0
    sigma_theta=0.15

    w_slow=0.0; w_fast=0.0
    alpha_slow=1.0; alpha_fast=1.0

    if isRunningOnArlo():
        arlo=CalibratedRobot()

    world=np.zeros((500,500,3),dtype=np.uint8)
    draw_world(est_pose, particles, world)

    print("Opening and initializing camera")
    if isRunningOnArlo():
        cam=camera.Camera(1, robottype='arlo', useCaptureThread=False)
        # IMPORTANT: typical Arlo kits: turn_angle(+deg) = CW/right -> True
        pathing=LocalizationPathing(arlo, cam, landmarkIDs, step_cm=15.0, rotation_deg=18.0, robot_cw_positive=True)
        stabilization_counter=0
        goal_hits=0
    else:
        cam=camera.Camera(0, robottype='macbookpro', useCaptureThread=False)

    last_dist_to_center = None

    while True:
        if cv2.waitKey(10)==ord('q'):
            break

        # Behaviour policy
        if isRunningOnArlo():
            if not pathing.seen_all_landmarks():
                drive = random.random() < (1/18)
                distance, angle = pathing.explore_step(drive)
            else:
                if stabilization_counter < 2:
                    stabilization_counter += 1
                    distance, angle = 0.0, 0.0
                else:
                    # drive toward known world midpoint
                    distance, angle = pathing.move_towards_goal_step(est_pose, center)

        # Update PF with commanded motion
        sample_motion_model(particles, distance, angle, sigma_d, sigma_theta)

        # Perception
        colour=cam.get_next_frame()
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if not isinstance(objectIDs, type(None)):
            objectIDs, dists, angles = filter_landmarks_by_distance(objectIDs,dists,angles)
            for i in range(len(objectIDs)):
                print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])

            measurement_model(particles, objectIDs, dists, angles, sigma_d, sigma_theta)

            weights=np.array([p.getWeight() for p in particles],dtype=np.float64)
            w_sum=float(weights.sum())
            if w_sum<=0.0 or not np.isfinite(w_sum):
                weights[:]=1.0/len(weights)
            else:
                weights/=w_sum

            w_avg=float(weights.mean())
            w_slow+=alpha_slow*(w_avg-w_slow)
            w_fast+=alpha_fast*(w_avg-w_fast)

            particles=resample_particles(particles, weights, w_fast, w_slow)
            cam.draw_aruco_objects(colour)
        else:
            for p in particles:
                p.setWeight(1.0/num_particles)

        est_pose=particle.estimate_pose(particles)

        # Stop when truly at midpoint (PF distance check with confirmation)
        dx=float(center[0]-est_pose.getX())
        dy=float(center[1]-est_pose.getY())
        dist_to_center=float(math.hypot(dx,dy))

        # simple overshoot protection: if getting worse, shrink step size
        if last_dist_to_center is not None and dist_to_center > last_dist_to_center + 5.0:
            # reduce step a bit if we overshot or diverged
            pathing.step_cm = max(8.0, pathing.step_cm * 0.7)
        last_dist_to_center = dist_to_center

        if dist_to_center <= CENTER_TOL_CM:
            goal_hits += 1
            if goal_hits >= GOAL_HITS_REQUIRED:
                print("[INFO] Reached midpoint. Stopping.")
                break
        else:
            goal_hits = 0

        if showGUI:
            draw_world(est_pose, particles, world)
            cv2.imshow("Robot view", colour)
            cv2.imshow("World view", world)

finally:
    cv2.destroyAllWindows()
    try:
        cam.terminateCaptureThread()
    except:
        pass
