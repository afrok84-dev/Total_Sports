

import cv2
import numpy as np

from Total_Sports.tennis_2D_transformation.services.court_corners_service import CourtCornersService



class TennisCourt:
    def __init__(
        self,
        image_width: int,
        image_height: int,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.court_mask = None
        self.corners = None
        
    def update_court_mask(self, mask: np.ndarray):
        self.court_mask = mask
        
    def get_court_size_in_meters(self) -> tuple[float, float]:
        # method to get court size in meters
        # size should be known and will be queryed from a database. Hardcoded for now.
        return (23.77, 10.97)  # Standard tennis court dimensions (length, width) in meters
    
    def get_corners_from_image(self) -> np.ndarray:
        if self.court_mask is None:
            raise ValueError("Court mask is not set.")
        self.corners = CourtCornersService(
            tennis_field_points=self.court_mask,
            image_width=self.image_width,
            image_height=self.image_height,
        ).get_field_corners()
        return self.corners

    def get_corners_real_world(self) -> np.ndarray:
        if self.corners is None:
            raise ValueError("Court corners have not been computed.")
        court_length_m, court_width_m = self.get_court_size_in_meters()
        real_world_corners = np.array([
            [0.0, court_length_m],   # TL
            [court_width_m,   court_length_m],   # TR
            [court_width_m,   0.0], # BR
            [0.0, 0.0], # BL
        ])
        return real_world_corners

    def get_homography_image_to_world(self) -> np.ndarray:
        if self.corners is None:
            raise ValueError("Court corners have not been computed.")
        real_world_corners = self.get_corners_real_world()
        H_img_to_world, _ = cv2.findHomography(self.corners, real_world_corners, method=0)
        return H_img_to_world
