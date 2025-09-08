import time
import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CalibratedRobot

calArlo = CalibratedRobot.CalibratedRobot()
FILENAME = "sonar_measurements.csv"

def measure_distance(actual_distance, samples=5):
    """Take multiple sonar readings at a known distance and save to CSV."""
    readings = []
    for i in range(samples):
        val = calArlo.arlo.read_front_ping_sensor()
        print(f"Sample {i+1}: {val} mm")
        readings.append(val)
        time.sleep(0.2)

measure_distance(100)

