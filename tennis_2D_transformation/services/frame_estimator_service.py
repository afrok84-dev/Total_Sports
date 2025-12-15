import numpy as np
import cv2
from collections import deque

def perspective_map_point(u: float, v: float, H_img_to_world: np.ndarray) -> np.ndarray:
    p = np.array([[[u, v]]], dtype=np.float32)
    w = cv2.perspectiveTransform(p, H_img_to_world)[0, 0]  # (x,y)
    return w.astype(np.float64)

def mad_gate(points_xy: np.ndarray, k: float = 3.5) -> np.ndarray:
    if len(points_xy) < 3:
        return np.ones(len(points_xy), dtype=bool)
    med = np.median(points_xy, axis=0)
    dev = np.linalg.norm(points_xy - med, axis=1)
    mad = np.median(np.abs(dev - np.median(dev))) + 1e-9
    z = 0.6745 * (dev - np.median(dev)) / mad
    return np.abs(z) < k

def fit_constant_velocity(t: np.ndarray, xy: np.ndarray):
    A = np.column_stack([np.ones_like(t), t])  # [1, t]
    ax, bx = np.linalg.lstsq(A, xy[:, 0], rcond=None)[0]
    ay, by = np.linalg.lstsq(A, xy[:, 1], rcond=None)[0]
    a = np.array([ax, ay], dtype=np.float64)
    b = np.array([bx, by], dtype=np.float64)
    return a, b

class FrameEstimatorService:
    def __init__(self, H_img_to_world: np.ndarray, window: int = 5):
        self.H = H_img_to_world.astype(np.float64)
        self.obs = deque(maxlen=window)  # (t, x, y, score)

    def add_observation_uv(self, u: float, v: float, t: float, score: float = 1.0):
        xy = perspective_map_point(u, v, self.H)
        self.obs.append((t, xy[0], xy[1], score))

    def estimate(self, t_query: float):
        if len(self.obs) < 3:
            return None

        arr = np.array(self.obs, dtype=np.float64)  # t,x,y,score
        t = arr[:, 0]
        xy = arr[:, 1:3]

        inliers = mad_gate(xy)
        t_in, xy_in = t[inliers], xy[inliers]
        if len(xy_in) < 3:
            return None

        # Fit constant velocity in a tiny window
        a, b = fit_constant_velocity(t_in - t_in[0], xy_in)

        dt = float(t_query - t_in[0])
        est_xy = a + b * dt
        return float(est_xy[0]), float(est_xy[1])
