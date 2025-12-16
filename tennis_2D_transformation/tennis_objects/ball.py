import cv2
import numpy as np
from collections import deque

from Total_Sports.tennis_2D_transformation.services.frame_estimator_service import FrameEstimatorService, perspective_map_point



class TennisBall:
    def __init__(self, mask: np.ndarray | None = None, speed_window=10):
        self.mask = mask
        self.ball_position = None  # (x, y) in image coordinates
        self.ball_positions_history = []
        self._world_hist = deque(maxlen=speed_window)  # (t, x, y)
        self._speed = 0.0
        
    def update_ball_mask(self, mask: np.ndarray, t: float, H_img_to_world: np.ndarray):
        self.mask = mask
        self.ball_position = self._centroid_from_pixels(mask)
        self.ball_positions_history.append(self.ball_position)
        if self.ball_position is None:
            return

        u, v = self.ball_position
        x, y = perspective_map_point(u, v, H_img_to_world)
        self._world_hist.append((t, float(x), float(y)))

        if len(self._world_hist) >= 2:
            t1, x1, y1 = self._world_hist[-2]
            t2, x2, y2 = self._world_hist[-1]
            dt = (t2 - t1)
            if dt > 1e-6:
                self._speed = float(np.hypot(x2 - x1, y2 - y1) / dt)

    def _centroid_from_mask(self, mask: np.ndarray | None):
        if mask is None:
            return None
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if mask.max() == 1:
            mask = mask * 255

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None

        # centroid
        u = float(xs.mean())
        v = float(ys.mean())
        return (u, v)

    def _centroid_from_pixels(self, pixels_xy: np.ndarray):
        return float(pixels_xy[:,0].mean()), float(pixels_xy[:,1].mean())

    def get_ball_position_in_relation_to_court(self, H_img_to_world: np.ndarray, court_bounds) -> tuple[float, float] | None:
        pos = self.get_ball_coordinates_in_real_world(H_img_to_world)
        if pos is None:
            return None
        x, y, _ = pos
        xmin, xmax, ymin, ymax = court_bounds
        rx = (x - xmin) / (xmax - xmin + 1e-9)
        ry = (y - ymin) / (ymax - ymin + 1e-9)
        return float(rx), float(ry)
    
    def get_ball_position_in_relation_to_players(self, player_positions: list[tuple[float, float]]) -> list[float]:
        # method to get ball position relative to player positions
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_ball_coordinates_in_frame(self) -> tuple[float, float] | None:
        return self.ball_position
    
    def get_ball_coordinates_in_real_world(self, H_img_to_world) -> tuple[float, float, float] | None:
        if self.ball_position is None:
            return None
        u, v = self.ball_position
        x, y = perspective_map_point(u, v, H_img_to_world)
        # You don't have z yet → return 0.0 for now or None
        return float(x), float(y), 0.0
    
    @property
    def ball_speed(self) -> float:
        return self._speed
    
    @property
    def is_ball_in_play(self) -> bool:
        # method to determine if current state of the ball is in play
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    @property
    def is_ball_out(self) -> bool:
        # method to determine if ball is out of play
        raise NotImplementedError("This method should be implemented by subclasses.")

    @property
    def is_serve(self) -> bool:
        # method to determine if the ball is in a serve state
        raise NotImplementedError("This method should be implemented by subclasses.")