import time
import robot

arlo = robot.Robot()

SPEED  = 64   
TRANSLATION_TIME = 2.5  
TURN_TIME  = 0.89  

CAL_KL = 0.980  
CAL_KR = 1.000   
MIN_PWR = 40   
MAX_PWR = 127

FORWARD = 1
BACKWARD = 0

def clamp_power(p):
    return max(MIN_PWR, min(MAX_PWR, int(round(p))))

def drive(speed, secs, dirLeft, dirRight):
    l = clamp_power(speed * CAL_KL) if speed > 0 else 0
    r = clamp_power(speed * CAL_KR) if speed > 0 else 0
    arlo.go_diff(l, r, dirLeft, dirRight)
    time.sleep(secs)
    arlo.stop()
    time.sleep(0.2) 
    
def drive_in_square(times=1):
    for _ in range(times):
        for _ in range(4):
            #Drive forwards   
            drive(SPEED, TRANSLATION_TIME, FORWARD, FORWARD)
            #Turn 90 degrees   
            drive(SPEED, TURN_TIME, BACKWARD, FORWARD)    
            
try:
    drive_in_square(3) 
finally:
    arlo.stop() 
            
   


    
