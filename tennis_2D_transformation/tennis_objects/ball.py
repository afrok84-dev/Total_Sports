

import cv2
import numpy as np

from services.frame_estimator_service import FrameEstimatorService


class TennisBall:
    def __init__(self, detection_mask: np.ndarray | None = None):
        self.detection_mask = detection_mask
        self.ball_position = None  # (x, y) in image coordinates
        
    def get_ball_position_in_relation_to_court(self) -> tuple[float, float]:
        # method to get ball position relative to court dimensions
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_ball_position_in_relation_to_players(self, player_positions: list[tuple[float, float]]) -> list[float]:
        # method to get ball position relative to player positions
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_ball_coordinates_in_frame(self, H_img_to_world) -> tuple[float, float]:
        fps = int(214/7) # 214 frames over 7 seconds
        estimator_service = FrameEstimatorService(H_img_to_world, window=5)

        for i, mask_text in enumerate(masks_as_text):
            t = i / fps

            pts = parse_sam_points(mask_text)
            res = sam_mask_to_center(pts)  # (u,v,score) or None

            if res is not None:
                u, v, score = res
                estimator_service.add_observation_uv(u, v, t=t, score=score)

            xy_hat = estimator_service.estimate(t_query=t)
            print(i, "ball topdown (x,y):", xy_hat)
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_ball_coordinates_in_real_world(self) -> tuple[float, float, float]:
        # method to get ball coordinates in real-world 3D space
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    @property
    def ball_speed(self) -> float:
        raise NotImplementedError("This method should be implemented by subclasses.")
    
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