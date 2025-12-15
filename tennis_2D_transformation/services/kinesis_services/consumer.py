import time
import re
import numpy as np


FRAME_RE = re.compile(r"^FRAME\s+(\d+)\s*$")

def iter_frames_from_mask_txt(path: str):
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


def simulate_consumption(path: str, width: int, height: int, fps: float = 30.0, loop: bool = False):
    """
    Simulates consuming frames in time order.
    - width/height should match the resized frame dimensions used when you produced the masks.
    """
    delay = 1.0 / fps

    while True:
        for frame_idx, pixels in iter_frames_from_mask_txt(path):
            ball_mask = pixels_to_mask(pixels, width=width, height=height)
            # "Consume" the frame (replace with your real processing)
            print(f"[consume] frame={frame_idx} pixels={len(pixels)} mask_sum={int(ball_mask.sum())}")
            # simulate realtime
            time.sleep(delay)

        if not loop:
            break


if __name__ == "__main__":
    MASK_FILE_ROOT = "input_raw/"  
    # These must match the resized dimensions used when generating masks
    W, H = 640, 311  # change this to your actual resized frame size
    simulate_consumption(MASK_FILE_ROOT, width=W, height=H, fps=10.0, loop=False)
