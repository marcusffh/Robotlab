import time
import robot
import Exercise1.CalibratedRobot as CalibratedRobot

calArlo = CalibratedRobot.CalibratedRobot()
    
def drive_in_square(times=1, meters=1, speed=None):
    for _ in range(times):
        for _ in range(4):
            #Drive forwards   
            calArlo.drive_distance(meters, speed = speed)
            #Turn 90 degrees   
            calArlo.turn_angle(90, speed = speed)  
            
try:
    drive_in_square(3) 
finally:
    calArlo.stop() 
            