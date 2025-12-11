
import numpy as np
from filterpy.kalman import KalmanFilter

class BallTracker:
    def __init__(self):
        self.kf = self.init_kf()

    def init_kf(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.zeros(4)
        kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        kf.H = np.array([[1,0,0,0],[0,1,0,0]])
        kf.P *= 20
        kf.R *= 10
        kf.Q *= 0.01
        return kf

    def update(self, x, y):
        self.kf.predict()
        self.kf.update(np.array([x,y]))
        return self.kf.x[:2]
