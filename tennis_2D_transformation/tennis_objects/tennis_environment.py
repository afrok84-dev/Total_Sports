import numpy as np
from Total_Sports.tennis_2D_transformation.tennis_objects.ball import TennisBall
from Total_Sports.tennis_2D_transformation.tennis_objects.court import TennisCourt
from Total_Sports.tennis_2D_transformation.input_masks.ball_detection_mask import tennis_ball
from Total_Sports.tennis_2D_transformation.input_masks.player1_detection_mask import tennis_player_1
from Total_Sports.tennis_2D_transformation.input_masks.player2_detection_mask import tennis_player_2
from Total_Sports.tennis_2D_transformation.input_masks.player3_detection_mask import tennis_player_3
from Total_Sports.tennis_2D_transformation.input_masks.player4_detection_mask import tennis_player_4
from Total_Sports.tennis_2D_transformation.input_masks.court_detection_mask import tennis_court
from Total_Sports.tennis_2D_transformation.utils.ball_missing_frames_filler import fill_gaps_linear_max_gap


class GeneralEnvironment:
    def __init__(
        self,
        match_id: str = "test_match",
        image_width: int = 640, # resized width in sam3 code
        image_height: int = 311, # resized height in sam3 code
    ):
        self.match_id = match_id
        self.image_width = image_width
        self.image_height = image_height
        self.court_object = TennisCourt(image_height=self.image_height, image_width=self.image_width)
        self.ball_object = TennisBall()
        self.tennis_players_object = None # skip for now

    def get_player_masks(self) -> list[np.ndarray]:
        # method that should consume player detection masks. Hardcoded for now.
        player1_points = np.fromstring(tennis_player_1.replace('[','').replace(']',''), sep=' ').reshape(-1, 2)
        player2_points = np.fromstring(tennis_player_2.replace('[','').replace(']',''), sep=' ').reshape(-1, 2)
        player3_points = np.fromstring(tennis_player_3.replace('[','').replace(']',''), sep=' ').reshape(-1, 2)
        player4_points = np.fromstring(tennis_player_4.replace('[','').replace(']',''), sep=' ').reshape(-1, 2)
        return [player1_points, player2_points, player3_points, player4_points]
    
    def frame_process(
        self,
        frame_idx: int,
        updated_court_mask: np.ndarray,
        updated_ball_mask: np.ndarray,
        updated_player_masks: list[np.ndarray],
        missing_ball_frames: bool = False,
        fps: int = 30,
    ):
        # this should be called on every frame update
        self.court_object.update_court_mask(updated_court_mask)
        self.court_object.get_corners_from_image()
        homography = self.court_object.get_homography_image_to_world()
        if missing_ball_frames:
            ball_poss_filled = fill_gaps_linear_max_gap(self.ball_object.ball_positions_history,)
            self.ball_object.ball_positions_history = ball_poss_filled
        else:
            self.ball_object.update_ball_mask(updated_ball_mask, frame_idx / fps, homography)
