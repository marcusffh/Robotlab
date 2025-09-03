import time
import robot

arlo = robot.Robot()

TRANSLATION_TIME = 2.5  
TURN_TIME  = 0.91  

CAL_KL = 0.980  
CAL_KR = 1.000   
MIN_PWR = 40   
MAX_PWR = 127

FORWARD = 1
BACKWARD = 0

speed  = 64   

def clamp_power(p):
    return max(MIN_PWR, min(MAX_PWR, int(round(p))))

def drive(speed, duration, leftDir, rightDir):
    l = clamp_power(speed * CAL_KL) if speed > 0 else 0
    r = clamp_power(speed * CAL_KR) if speed > 0 else 0
    arlo.go_diff(l, r, leftDir, rightDir)
    time.sleep(duration)
    arlo.stop()
    time.sleep(0.2) 
    
def drive_in_square(times=1):
    for _ in range(times):
        for _ in range(4):
            #Drive forwards   
            drive(speed, TRANSLATION_TIME, FORWARD, FORWARD)
            #Turn 90 degrees   
            drive(speed, TURN_TIME, BACKWARD, FORWARD)    
            
try:
    drive_in_square(1) 
finally:
    arlo.stop() 
            
   


    
