
import numpy as np

class BallPhysics:
    def detect_bounce(self, trajectory):
        # placeholder logic
        # bounce = sudden sign change in vertical velocity
        bounces = []
        for i in range(2, len(trajectory)):
            prev = trajectory[i-1]
            curr = trajectory[i]
            if curr[1] > prev[1]:  # simplistic
                bounces.append(i)
        return bounces
