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


def analyze_measurements():
    """Compute standard deviation of error vs distance, and plot measured vs actual."""
    # Your manual data
    data = {
        "actual_distance": [100]*5 + [500]*5 + [1000]*5 + [2000]*5 + [3000]*5,
        "measured": [
            108, 107, 107, 107, 108, # 100mm
            511, 508, 507, 507, 507,   # 500 mm
            997, 998, 998, 999, 999,   # 1000 mm
            2000, 2000, 2001, 2001, 2001,  # 2000 mm
            2982, 2982, 2982, 2982, 2982   # 3000 mm
        ]
    }
    df = pd.DataFrame(data)
    
    # Error = measured - actual
    df["error"] = df["measured"] - df["actual_distance"]

    # Group by actual distance to compute stats
    grouped = df.groupby("actual_distance").agg(
        mean_measured=("measured", "mean"),
        std_error=("error", "std")
    ).reset_index()

    print("\n Measurement Summary:")
    print(grouped)

    # Plot measured vs actual
    plt.figure(figsize=(7,5))
    plt.scatter(df["actual_distance"], df["measured"], alpha=0.6, label="Measurements")
    plt.plot(df["actual_distance"], df["actual_distance"], "r--", label="Ideal Linear")
    plt.xlabel("Actual Distance (mm)")
    plt.ylabel("Measured Distance (mm)")
    plt.title("Sonar Measurements vs Actual Distance")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot std of error as a function of distance
    plt.figure(figsize=(7,5))
    plt.plot(grouped["actual_distance"], grouped["std_error"], marker="o")
    plt.xlabel("Actual Distance (mm)")
    plt.ylabel("Std of Measurement Error (mm)")
    plt.title("Precision of Sonar vs Distance")
    plt.grid(True)
    plt.show()


analyze_measurements()


