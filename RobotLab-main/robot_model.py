import numpy as np

class RobotModel:
    def __init__(self, robot_radius = 0.25):
        self.robot_radius = robot_radius
        
    def forward(self, from_pos, to_pos, n_steps):
        from_pos = np.array(from_pos)
        to_pos = np.array(to_pos)
        return [(from_pos + (to_pos - from_pos) * (i+1)/n_steps).tolist() for i in range(n_steps)]

