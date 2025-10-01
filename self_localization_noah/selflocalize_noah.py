#!/usr/bin/env python3
"""
selflocalize_noah.py — Exercise 5 with CalibratedRobot
Shows two windows:
  - "Camera": PiCam feed with ArUco detections
  - "PF": particle filter map (landmarks, particles, estimate)
"""

import math, time, numpy as np, cv2
from Robotutils.CalibratedRobot import CalibratedRobot
from .camera_noah import LandmarkCamera, draw_detections
from .particle_noah import (
    init_particles, predict, weight, resample_systematic,
    estimate_pose, effective_sample_size
)

# ---------------- CONFIG ----------------
LANDMARKS = {
    9: (0.0, 0.0),
    11: (3.0, 0.0),
}
TARGET = ((LANDMARKS[9][0] + LANDMARKS[11][0]) / 2,
          (LANDMARKS[9][1] + LANDMARKS[11][1]) / 2)

N_PARTICLES   = 500
BOUNDS_XY     = ((-0.5, 3.5), (-1.0, 1.0))
THETA_RANGE   = (-math.pi, math.pi)
SIGMA_TRANS   = 0.15
SIGMA_ROT     = 0.15
SIGMA_R       = 0.07
SIGMA_B       = 0.05
RECOVERY_FRAC = 0.05

SCAN_STEP_DEG = 20.0
TURN_SPEED    = 50
DRIVE_SPEED   = 50
DRIVE_STEP_M  = 0.25
DIST_TOL      = 0.15
MAX_SECONDS   = 240
SCAN_SETTLE_TIME = 0.3

LOOP_HZ = 10.0
DT      = 1.0/LOOP_HZ

# ------------- Helpers ------------------
def angle_wrap(a): return (a+math.pi)%(2*math.pi)-math.pi

def commanded_predict_turn(particles, arlo, angle_deg, speed):
    angle_rad = math.radians(angle_deg)
    duration = arlo.TURN_TIME * (abs(angle_deg)/90.0) * (arlo.default_speed/float(speed))
    omega = angle_rad/duration if duration>1e-6 else 0.0
    predict(particles, 0, omega, duration, SIGMA_TRANS, SIGMA_ROT)

def commanded_predict_drive(particles, arlo, meters, speed):
    duration = arlo.TRANSLATION_TIME * meters * (arlo.default_speed/float(speed))
    v = meters/duration if duration>1e-6 else 0.0
    predict(particles, v, 0, duration, SIGMA_TRANS, SIGMA_ROT)

def pf_update_in_place(particles, cam, steps=5):
    weights = np.ones(len(particles))/len(particles)
    for _ in range(steps):
        predict(particles, 0,0,DT,SIGMA_TRANS,SIGMA_ROT)
        dets = cam.read()
        weights = weight(particles,dets,LANDMARKS,SIGMA_R,SIGMA_B)
        if effective_sample_size(weights)<0.5*len(particles):
            particles[:] = resample_systematic(particles,weights,RECOVERY_FRAC,BOUNDS_XY,THETA_RANGE)
            weights = np.ones(len(particles))/len(particles)
        frame = cam._grab_frame() if hasattr(cam,"_grab_frame") else None
        frame_vis = draw_detections(frame,dets)
        if frame_vis is not None:
            cv2.imshow("Camera", frame_vis)
        time.sleep(DT)
    return weights

def show_particles(particles, est_pose, window="PF"):
    scale, size = 150, 600
    img = 255*np.ones((size,size,3),np.uint8)
    for lid,(lx,ly) in LANDMARKS.items():
        cx,cy=int(lx*scale+size/2),int(size/2-ly*scale)
        cv2.circle(img,(cx,cy),6,(0,0,255),-1)
        cv2.putText(img,str(lid),(cx+5,cy-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
    for px,py,pt in particles:
        cx,cy=int(px*scale+size/2),int(size/2-py*scale)
        cv2.circle(img,(cx,cy),1,(200,200,200),-1)
    ex,ey,et=est_pose
    cx,cy=int(ex*scale+size/2),int(size/2-ey*scale)
    cv2.circle(img,(cx,cy),5,(0,255,0),-1)
    dx,dy=int(15*math.cos(et)),int(-15*math.sin(et))
    cv2.arrowedLine(img,(cx,cy),(cx+dx,cy+dy),(0,255,0),2)
    cv2.imshow(window,img)
    cv2.waitKey(1)

# ------------- Phases -------------------
def phase_scan_until_seen_both(arlo, cam, particles):
    print("[SCAN] Starting scan...")
    seen=set(); start=time.time()
    while time.time()-start<60:
        arlo.turn_angle(SCAN_STEP_DEG,speed=TURN_SPEED)
        time.sleep(SCAN_SETTLE_TIME)
        commanded_predict_turn(particles,arlo,SCAN_STEP_DEG,TURN_SPEED)
        dets=cam.read()
        for (lid,_,_) in dets:
            if lid in LANDMARKS: seen.add(lid)
        w=weight(particles,dets,LANDMARKS,SIGMA_R,SIGMA_B)
        if effective_sample_size(w)<0.5*len(particles):
            particles[:]=resample_systematic(particles,w,RECOVERY_FRAC,BOUNDS_XY,THETA_RANGE)
            w=np.ones(len(particles))/len(particles)
        est=estimate_pose(particles,w)
        print(f"[SCAN] Seen={sorted(seen)} est=({est[0]:.2f},{est[1]:.2f},{est[2]:.2f})")
        frame=cam._grab_frame() if hasattr(cam,"_grab_frame") else None
        frame_vis=draw_detections(frame,dets)
        if frame_vis is not None: cv2.imshow("Camera",frame_vis)
        show_particles(particles,est)
        if all(l in seen for l in LANDMARKS):
            print("[SCAN] Both seen.")
            return True
    return False

def phase_drive_to_midpoint(arlo,cam,particles):
    print(f"[DRIVE] Target={TARGET}")
    t0=time.time()
    while time.time()-t0<MAX_SECONDS:
        w=pf_update_in_place(particles,cam,steps=int(0.5*LOOP_HZ))
        x,y,th=estimate_pose(particles,w)
        dx,dy=TARGET[0]-x,TARGET[1]-y
        dist=math.hypot(dx,dy)
        goal_heading=math.atan2(dy,dx)
        dth=angle_wrap(goal_heading-th)
        print(f"[DRIVE] pose=({x:.2f},{y:.2f},{th:.2f}) dist={dist:.2f} dth={dth:.2f}")
        show_particles(particles,(x,y,th))
        if dist<=DIST_TOL:
            print("[DRIVE] Reached midpoint ✅")
            return True
        turn_deg=math.degrees(dth)
        turn_deg=max(-60,min(60,turn_deg))
        if abs(turn_deg)>2:
            arlo.turn_angle(turn_deg,speed=TURN_SPEED)
            commanded_predict_turn(particles,arlo,turn_deg,TURN_SPEED)
            _=pf_update_in_place(particles,cam,steps=2)
        step_m=min(dist,DRIVE_STEP_M)
        if step_m>0.03:
            arlo.drive_distance(step_m,direction=arlo.FORWARD,speed=DRIVE_SPEED)
            commanded_predict_drive(particles,arlo,step_m,DRIVE_SPEED)
            _=pf_update_in_place(particles,cam,steps=2)
    return False

# ------------- Main ---------------------
def main():
    print("[MAIN] Self-localization with PF + Camera")
    arlo=CalibratedRobot()
    cam=LandmarkCamera(use_mock=False); cam.open()
    particles=init_particles(N_PARTICLES,BOUNDS_XY,THETA_RANGE)
    try:
        if not phase_scan_until_seen_both(arlo,cam,particles):
            print("[MAIN] Scan failed."); return
        ok=phase_drive_to_midpoint(arlo,cam,particles)
        print("[MAIN] Done" if ok else "[MAIN] Failed")
    finally:
        cam.close(); arlo.stop(); cv2.destroyAllWindows()

if __name__=="__main__": main()
