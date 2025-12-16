import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from Total_Sports.tennis_2D_transformation.tennis_objects.ball import TennisBall
import cv2

positions_hardcoded = [(332.07142857142856, 242.85714285714286), (332.07142857142856, 242.85714285714286), (326.4375, 246.625), (323.1333333333333, 248.93333333333334), (311.1111111111111, 260.8888888888889), (308.0, 264.63157894736844), (301.5, 273.0), (295.4, 282.9), (308.44, 216.92), (309.6818181818182, 210.86363636363637), (313.06666666666666, 198.33333333333334), (316.35, 187.55), (317.60869565217394, 182.47826086956522), (320.2352941176471, 172.94117647058823), (322.25, 163.66666666666666), (323.84615384615387, 160.53846153846155), (326.4, 153.4), (328.5, 146.71428571428572), (331.25, 138.58333333333334), (332.9166666666667, 133.66666666666666), (334.0, 131.5), (335.54545454545456, 127.63636363636364), (337.54545454545456, 124.36363636363636), (338.2, 123.0), (339.875, 120.875), (343.0, 116.28571428571429), (344.6666666666667, 115.11111111111111), (345.3333333333333, 114.88888888888889), (346.5, 114.8), (347.8888888888889, 114.88888888888889), (348.55555555555554, 114.66666666666667), (349.55555555555554, 114.66666666666667), (355.5, 102.0), (356.0, 95.5), (356.875, 91.875), (325.25, 116.33333333333333), (321.46666666666664, 122.66666666666667), (319.5, 126.5), (315.4375, 134.25), (300.95652173913044, 169.2608695652174), (294.6363636363636, 186.95454545454547), (283.9642857142857, 220.89285714285714), (277.46666666666664, 244.26666666666668), (268.75, 274.6666666666667), (267.3636363636364, 274.09090909090907), (264.15384615384613, 273.2307692307692), (260.84615384615387, 272.7692307692308), (259.5, 273.0), (256.38461538461536, 273.61538461538464), (253.0, 275.5), (251.5, 276.5), (248.46153846153845, 279.15384615384613), (245.3846153846154, 282.61538461538464), (243.6, 284.6), (240.5, 289.5), (237.41176470588235, 295.11764705882354), (235.52941176470588, 298.3529411764706), (242.41379310344828, 240.20689655172413), (244.44444444444446, 223.62962962962962), (245.56, 215.88), (249.5, 185.5), (250.58823529411765, 176.05882352941177), (252.0, 167.5), (253.0, 157.2), (254.0, 151.0), (255.5, 144.5), (255.9090909090909, 140.63636363636363), (256.0, 139.0), (256.75, 134.5), (257.0, 134.0), (257.14285714285717, 132.71428571428572), (257.125, 131.875), (257.6666666666667, 131.11111111111111), (261.0, 84.2), (263.0, 85.0), (285.6363636363636, 89.9090909090909), (290.3636363636364, 91.0909090909091), (295.2, 94.0), (297.375, 94.75), (319.54545454545456, 108.36363636363636), (327.14285714285717, 114.92857142857143), (339.92857142857144, 126.28571428571429), (361.0, 318.0), (361.0, 318.22222222222223), (373.4, 164.6), (361.1, 318.1), (391.75, 190.05), (397.3, 199.05), (403.35, 208.75), (412.75, 224.15), (419.0952380952381, 234.61904761904762), (422.3333333333333, 240.28571428571428), (455.3076923076923, 270.0), (459.1666666666667, 269.9166666666667), (461.6363636363636, 269.90909090909093), (466.0, 270.6666666666667), (470.5, 271.0), (472.5, 272.0), (477.3636363636364, 273.90909090909093), (482.2307692307692, 276.2307692307692), (484.5, 278.0), (489.0, 281.5), (482.51428571428573, 231.37142857142857), (473.2631578947368, 179.47368421052633), (472.0, 173.5), (466.5, 152.5), (464.5, 146.5), (462.5, 141.5), (459.6666666666667, 136.11111111111111), (458.0, 134.0), (457.09090909090907, 133.36363636363637), (455.125, 131.875), (453.42857142857144, 131.14285714285714), (452.2857142857143, 131.42857142857142), (451.0, 131.71428571428572), (445.6363636363636, 87.54545454545455), (444.0, 79.0)]



def plot_ball_image_space(ball: TennisBall, width: int, height: int, fps: float = 30.0):
    """
    positions: list of (u, v) or None, one per frame
    """
    # positions = ball.ball_positions_history
    positions = ball # due to hardcode
    
    
    # print("!!!!!!!!!!!!!!")
    # print(positions)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # invert y for image coords
    ax.set_title("Ball positions (image space)")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.grid(True)

    trail_line, = ax.plot([], [], "-o", markersize=2)
    current_point = ax.scatter([], [], s=60)

    xs, ys = [], []

    def init():
        trail_line.set_data([], [])
        current_point.set_offsets(np.empty((0, 2)))  # ✅ FIX
        return trail_line, current_point

    def update(frame_idx: int):
        pos = positions[frame_idx]
        if pos is not None:
            u, v = pos
            xs.append(u)
            ys.append(v)

        trail_line.set_data(xs, ys)

        if xs:
            current_point.set_offsets(np.array([[xs[-1], ys[-1]]], dtype=float))
        else:
            current_point.set_offsets(np.empty((0, 2)))

        ax.set_title(f"Ball positions (image space) — frame {frame_idx+1}/{len(positions)}")
        return trail_line, current_point

    interval_ms = int(1000 / fps)
    anim = FuncAnimation(
        fig,
        update,
        frames=len(positions),
        init_func=init,
        interval=interval_ms,
        blit=False,   # keep False to avoid the _resize_id issue
        repeat=False,
    )

    plt.show()
    return anim
    
    # fig, ax = plt.subplots(figsize=(7, 5))
    # ax.set_xlim(0, width)
    # ax.set_ylim(height, 0)  # invert y for image coords
    # ax.set_title("Ball position (image space)")
    # ax.set_xlabel("x (pixels)")
    # ax.set_ylabel("y (pixels)")
    # ax.grid(True)

    # current_point = ax.scatter([], [], s=60)

    # def init():
    #     current_point.set_offsets(np.empty((0, 2)))
    #     return (current_point,)

    # def update(frame_idx: int):
    #     pos = positions[frame_idx]
    #     if pos is not None:
    #         u, v = pos
    #         current_point.set_offsets(np.array([[u, v]], dtype=float))
    #     else:
    #         current_point.set_offsets(np.empty((0, 2)))

    #     ax.set_title(f"Ball position (image space) — frame {frame_idx+1}/{len(positions)}")
    #     return (current_point,)

    # interval_ms = int(1000 / fps)
    # anim = FuncAnimation(
    #     fig,
    #     update,
    #     frames=len(positions),
    #     init_func=init,
    #     interval=interval_ms,
    #     blit=False,
    #     repeat=False,
    # )

    # plt.show()
    # return anim


def animate_ball_over_video(
    video_path: str,
    positions: list[tuple[float, float] | None],
    fps: float | None = None,
    resize_to: tuple[int, int] | None = None,  # (width, height) if you want to force match your coords
    start_frame: int = 0,
    max_frames: int | None = None,
):
    """
    positions: list of (u,v) in image pixel coords (same coord system as the video frames) or None
    resize_to: if your positions were computed on resized frames, resize the video frames to match
    """
    cap = cv2.VideoCapture(video_path)
    print("CAP_PROP_FPS:", cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval_ms = int(1000 / (fps or video_fps))

    # Jump to start frame if requested
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read first frame to initialize the plot
    ok, frame_bgr = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame from video.")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if resize_to is not None:
        w, h = resize_to
        frame_rgb = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_AREA)
    else:
        h, w = frame_rgb.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Ball over video")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # image coords
    ax.axis("off")

    im = ax.imshow(frame_rgb)
    ball_scatter = ax.scatter([], [], s=60)

    # Ensure scatter offset “empty” is correct shape
    EMPTY_OFFSETS = np.empty((0, 2), dtype=float)

    # How many frames to animate
    total = len(positions)
    if max_frames is not None:
        total = min(total, max_frames)

    # We already consumed 1 frame for init; we’ll treat it as frame 0 of the animation
    frame_index = 0

    def init():
        ball_scatter.set_offsets(EMPTY_OFFSETS)
        return (im, ball_scatter)

    import time

    t0 = None

    def update(_):
        nonlocal frame_index, t0

        if t0 is None:
            t0 = time.perf_counter()

        # which frame should we be on?
        elapsed = time.perf_counter() - t0
        target = int(elapsed * (fps or video_fps))  # frame number since start

        if target <= frame_index:
            return (im, ball_scatter)

        # skip ahead (drop frames) to catch up
        skip = target - frame_index
        for _ in range(skip):
            ok, frame_bgr = cap.read()
            if not ok:
                return (im, ball_scatter)
            frame_index += 1

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if resize_to is not None:
            frame_rgb = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_AREA)
        im.set_data(frame_rgb)

        pos = positions[frame_index] if frame_index < len(positions) else None
        if pos is None:
            ball_scatter.set_offsets(EMPTY_OFFSETS)
        else:
            u, v = pos
            ball_scatter.set_offsets(np.array([[u, v]], dtype=float))

        return (im, ball_scatter)

    anim = FuncAnimation(
        fig,
        update,
        frames=total,
        init_func=init,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    plt.show()
    cap.release()
    return anim



plot_ball_image_space(positions_hardcoded, width=640, height=311, fps=30.0)  # for testing

# video_path = r"C:\Users\User\Desktop\tennis_afro_gabi_project\Total_Sports\tennis_tracking_project\input\input_video.mp4"
# animate_ball_over_video(
#     video_path=video_path,
#     positions=positions_hardcoded,
#     fps=30.0,
#     resize_to=(640, 311),
#     start_frame=0,
# )