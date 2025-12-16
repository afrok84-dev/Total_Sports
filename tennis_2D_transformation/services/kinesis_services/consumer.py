import time
import re
import numpy as np
from pathlib import Path

from Total_Sports.tennis_2D_transformation.tennis_objects.tennis_environment import GeneralEnvironment
from Total_Sports.tennis_2D_transformation.utils.ball_plot import plot_ball_image_space


# from tennis_2D_transformation.tennis_objects.tennis_environment import GeneralEnvironment


FRAME_RE = re.compile(r"^FRAME\s+(\d+)\s*$")

def iter_frames_from_mask_txt(path: Path):
    """
    Yields (frame_idx, pixels_np) where pixels_np is shape (N,2) with columns [x,y].
    """
    current_frame = None
    current_pixels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            m = FRAME_RE.match(line)
            if m:
                # flush previous frame
                if current_frame is not None:
                    yield current_frame, np.array(current_pixels, dtype=np.int32)
                current_frame = int(m.group(1))
                current_pixels = []
                continue

            # pixel line: "x y"
            parts = line.split()
            if len(parts) == 2:
                x, y = int(parts[0]), int(parts[1])
                current_pixels.append((x, y))

    # flush last frame
    if current_frame is not None:
        yield current_frame, np.array(current_pixels, dtype=np.int32)


def pixels_to_mask(pixels_xy: np.ndarray, width: int, height: int):
    """
    pixels_xy: (N,2) array with columns [x,y]
    returns: (height,width) bool mask
    """
    # mask = np.zeros((height, width), dtype=bool)
    # if pixels_xy.size == 0:
    #     return mask
    # xs = pixels_xy[:, 0]
    # ys = pixels_xy[:, 1]
    # valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    # mask[ys[valid], xs[valid]] = True
    # return mask
    return pixels_xy.reshape(-1, 2)


def simulate_consumption(root_path: Path, width: int, height: int, fps: float = 30.0, loop: bool = False):
    """
    Simulates consuming frames in time order.
    - width/height should match the resized frame dimensions used when you produced the masks.
    """
    TOTAL_FRAMES = 214
    delay = 1.0 / fps
    ball_generator = iter_frames_from_mask_txt(root_path / "tennis_ball_object_000.txt")
    
    court_generator = iter_frames_from_mask_txt(root_path / "tennis_court_object_000.txt")

    ball_frame_id, ball_frame_pixels = next(ball_generator)
    court_frame_id, court_frame_pixels = next(court_generator)
    missing_ball_frames = False
    
    ball_mask = pixels_to_mask(ball_frame_pixels, width=width, height=height)
    court_mask = pixels_to_mask(court_frame_pixels, width=width, height=height)
    tennis_env = GeneralEnvironment(image_width=width, image_height=height)
    tennis_env.frame_process(
        frame_idx=ball_frame_id,
        updated_court_mask=court_mask,
        updated_ball_mask=ball_mask,
        updated_player_masks=[],
    )

    for frame_idx in range(1, TOTAL_FRAMES + 1):
        try:
            while ball_frame_id < frame_idx:
                ball_frame_id, ball_frame_pixels = next(ball_generator)

            if ball_frame_id == frame_idx:
                missing_ball_frames = False
                ball_mask = pixels_to_mask(ball_frame_pixels, width=width, height=height)
            else:
                missing_ball_frames = True
                # TODO: handle missing ball data better
                print(f"[consume] frame={frame_idx} NO BALL DATA")

            while court_frame_id < frame_idx:
                court_frame_id, court_frame_pixels = next(court_generator)

            if court_frame_id == frame_idx:
                court_mask = pixels_to_mask(court_frame_pixels, width=width, height=height)
            else:
                # TODO: handle missing court data better
                print(f"[consume] frame={frame_idx} NO COURT DATA")

            tennis_env.frame_process(
                frame_idx=frame_idx,
                updated_court_mask=court_mask,
                updated_ball_mask=ball_mask,
                updated_player_masks=[],
                missing_ball_frames=missing_ball_frames,
            )
            print(f"[consume] frame={frame_idx} BALL frame: {ball_frame_id} COURT frame: {court_frame_id}")
        except Exception as e:
            print(f"[consume] frame={frame_idx} ERROR: {e}")
            continue
    
    plot_ball_image_space(tennis_env.ball_object, width=width, height=height)


if __name__ == "__main__":
    INPUT_DIR = Path(__file__).resolve().parents[3]   # .../tennis_2D_transformation
    # If your txt files are in tennis_2D_transformation/input_masks:
    MASK_DIR = INPUT_DIR / "tennis_2D_transformation/input_raw"
    MASK_FILE_ROOT = "Total_Sports/tennis_2D_transformation/input_raw"  
    # These must match the resized dimensions used when generating masks
    W, H = 640, 311  # change this to your actual resized frame size
    simulate_consumption(MASK_DIR, width=W, height=H, fps=10.0, loop=False)
