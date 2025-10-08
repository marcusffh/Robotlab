import numpy as np

class VectorUtils:
    @staticmethod
    def rotate_vector(v, angle):
        """Rotate 2D vector v by angle (radians)"""
        c, s = np.cos(angle), np.sin(angle)
        x_new = c*v[0] - s*v[1]
        y_new = s*v[0] + c*v[1]
        return np.array([x_new, y_new])
