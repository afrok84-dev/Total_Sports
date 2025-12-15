import cv2
from matplotlib import pyplot as plt
import numpy as np
from Total_Sports.tennis_2D_transformation.services.court_corners_service import Court2DService
from input_masks.ball_detection_mask import mask_tennis_ball
from input_masks.player1_detection_mask import mask_tennis_player_1
from input_masks.player2_detection_mask import mask_tennis_player_2
from input_masks.player3_detection_mask import mask_tennis_player_3
from input_masks.player4_detection_mask import mask_tennis_player_4
from input_masks.court_detection_mask import mask_tennis_court

class TennisPlotter:
    def __init__(
        self,
        ball_points: np.ndarray[tuple[int, int], np.dtype[np.float64]], 
        tennis_field_points: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        tennis_players_points: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
        image_width: int,
        image_height: int,
        court_length_m: float = 23.77,
        court_width_m: float = 10.97,
    ):
        self.court_length_m = court_length_m
        self.court_width_m = court_width_m
        self.image_width = image_width
        self.image_height = image_height
        self.ball_points = ball_points
        self.tennis_field_points = tennis_field_points
        self.tennis_players_points = tennis_players_points
        
        self.court_service = Court2DService(
            tennis_field_points=self.tennis_field_points,
            image_width=self.image_width,
            image_height=self.image_height,
        )
        self.field_corners = self.court_service.get_field_corners()
        

    def plot_detection_coordinates(self):
        plt.scatter(self.tennis_field_points[:, 0], self.tennis_field_points[:, 1],
            color='blue', s=5, label="Court Mask")
        
        for i, player_points in enumerate(self.tennis_players_points):
            color = 'red' if i % 2 == 0 else 'green'
            plt.scatter(player_points[:, 0], player_points[:, 1],
                color=color, s=20, label=f"Player {i+1}")
            
        plt.scatter(self.ball_points[:, 0], self.ball_points[:, 1],
                    color='yellow', s=40, label="Ball")
        plt.scatter(self.field_corners[:, 0], self.field_corners[:, 1],
                    color='magenta', s=80, marker='X', label="Court Corners")

        corner_names = ["TL", "TR", "BR", "BL"]
        for (x, y), name in zip(self.field_corners, corner_names):
            plt.text(x + 3, y - 3, name, color='magenta', fontsize=12, weight='bold')

        plt.xlim(0, self.image_width)
        plt.ylim(self.image_height, 0)
        plt.gca().set_aspect('equal')

        plt.title("Detected Points with Court Corners (Convex Hull)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_topdown_homography_single_points(
        self,
        out_size_px=(600, 1200),   # (W, H)
        pad_ratio=0.08,            # ~8% space around the court
        point_mode="last",         # "last" or "centroid"
        show_court_mask=False,     # keep False if you want only dots + lines
    ):
        W, Hh = out_size_px
        pad_x = int(W * pad_ratio)
        pad_y = int(Hh * pad_ratio)

        # src corners in image MUST be TL, TR, BR, BL
        src = self.field_corners.astype(np.float32)

        # dst corners in top-down plane, padded rectangle (TL, TR, BR, BL)
        dst = np.array([
            [pad_x,         pad_y],
            [W - 1 - pad_x, pad_y],
            [W - 1 - pad_x, Hh - 1 - pad_y],
            [pad_x,         Hh - 1 - pad_y],
        ], dtype=np.float32)

        H_img2top = cv2.getPerspectiveTransform(src, dst)

        # ----- collapse to single point per object -----
        ball_pt = self._pick_single_point(self.ball_points, mode=point_mode)
        player_pts = [self._pick_single_point(p, mode=point_mode) for p in self.tennis_players_points]

        # warp single points
        ball_td = self._warp_points(ball_pt[None, :], H_img2top)[0]
        players_td = [self._warp_points(pt[None, :], H_img2top)[0] for pt in player_pts]

        # (optional) warp court mask points only if you want them
        if show_court_mask:
            field_td = self._warp_points(self.tennis_field_points, H_img2top)

        # ----- plotting (black bg, white court lines, 1 dot each) -----
        fig, ax = plt.subplots()
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        # draw court border (padded rectangle)
        rect = np.array([
            [pad_x, pad_y],
            [W - 1 - pad_x, pad_y],
            [W - 1 - pad_x, Hh - 1 - pad_y],
            [pad_x, Hh - 1 - pad_y],
            [pad_x, pad_y],
        ], dtype=np.float32)
        ax.plot(rect[:, 0], rect[:, 1], linewidth=2)

        # Optional: show warped court mask points (usually not needed if you draw lines yourself)
        if show_court_mask:
            ax.scatter(field_td[:, 0], field_td[:, 1], s=1, alpha=0.35, label="Court Mask")

        # players: one point each
        for i, ptd in enumerate(players_td):
            color = "blue" if i % 2 == 0 else "blue"  # adjust if you want different player colors
            ax.scatter(ptd[0], ptd[1], s=60, color=color, edgecolors="none", label=f"Player {i+1}")

        # ball: one point
        ax.scatter(ball_td[0], ball_td[1], s=60, color="yellow", edgecolors="none", label="Ball")

        # axes / framing
        ax.set_aspect("equal")
        ax.set_xlim(0, W)
        ax.set_ylim(Hh, 0)  # invert y to look image-like
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # If you don’t want legend, comment these out
        # ax.legend(facecolor="black", edgecolor="white", labelcolor="white")

        plt.show()

    @staticmethod
    def _pick_single_point(points_xy: np.ndarray, mode: str = "last") -> np.ndarray:
        """
        Return one (x,y) from an Nx2 array.
        mode="last": use last point
        mode="centroid": use mean of points
        """
        if points_xy is None or len(points_xy) == 0:
            raise RuntimeError("No points provided to pick from")

        pts = np.asarray(points_xy, dtype=np.float32)

        if mode == "centroid":
            return pts.mean(axis=0)
        # default: last
        return pts[-1]

    @staticmethod
    def _warp_points(points_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        return warped

tennis_plotter = TennisPlotter(
    ball_points=mask_tennis_ball,
    tennis_field_points=mask_tennis_court,
    tennis_players_points=[mask_tennis_player_1, mask_tennis_player_2, mask_tennis_player_3, mask_tennis_player_4],
    image_width=640, # resized width in sam3 code
    image_height=311, # resized height in sam3 code
)

# tennis_plotter.plot_detection_coordinates()
tennis_plotter.plot_topdown_homography_single_points(
    out_size_px=(600, 1200),
    pad_ratio=0.2,
    point_mode="last",
    show_court_mask=False,
)
