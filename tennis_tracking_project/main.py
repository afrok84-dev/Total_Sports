import cv2
import numpy as np

from trackers.court_detection import CourtDetector
from detectors.yolov5_detector import YOLOv5Detector
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker
from analytics.court_homography import Homography
from analytics.ball_physics import BallPhysics
from Utilities.court_points import extract_points
from Utilities.tennis_court_image import court_2D_image
from Utilities.meters_to_radar import meters_to_radar
from Utilities.overlay_radar_transparent import overlay_radar_transparent
from Utilities.order_court_points import order_court_points
from Utilities.fig_to_numpy import fig_to_numpy

VIDEO_PATH = "input/input_video.mp4"
MODEL_PATH = "models/tennis_yolov5.pt"
OUTPUT_TRACKING_PATH = "output_tracking.mp4"

def main():
    # -------------------------
    # Initialize modules
    # -------------------------
    detector = YOLOv5Detector(MODEL_PATH)
    player_tracker = PlayerTracker()
    ball_tracker = BallTracker()
    court_detector = CourtDetector()
    homography = Homography()


    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_tracking = cv2.VideoWriter(
        OUTPUT_TRACKING_PATH, fourcc, fps, (frame_w, frame_h)
    )

    # -------------------------
    # Homography setup
    # -------------------------
    court_length, court_width = 23.77, 8.23                                # Tennis court dimensions in meters
    Radar_W, Radar_H = 450, 600                                            # Radar image size in pixels
    dst_corners = np.array([[0, 0],[court_width, 0],
                            [court_width, court_length],
                            [0, court_length]],
                              dtype=np.float32)

    homography_points_src = extract_points("./input/mask_coordinates.txt")
    homography_points_src = order_court_points(homography_points_src)
    
    homography.compute(homography_points_src, dst_corners)
    
    homography_points = [homography.warp_point(p) for p in homography_points_src]
    print("Homography court points (meters):", homography_points)
    # -------------------------
    # Create radar background
    # -------------------------
    fig = court_2D_image(homography_points)
    court_image = fig_to_numpy(fig)
    if court_image is None:
        raise RuntimeError("court_image.png not found")
    radar_bg = cv2.resize(court_image, (Radar_W, Radar_H))

    # -------------------------
    # Stable tennis players (Fix 4)
    # -------------------------
    selected_player_ids = set()

    frame_count = 0
    player_trails = {}

    # Main loop
    # -------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        tracking_frame = frame.copy()
        radar_frame = radar_bg.copy()

        # --- Detection + tracking ---
        detections = detector.detect(tracking_frame)
        player_dets = [d for d in detections if d["cls"] == 0]
        player_tracks = player_tracker.update(player_dets, tracking_frame)

        players_metric = []

        # --- Collect confirmed tracks ---
        for track in player_tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = track.to_ltrb()
            track_id = int(track.track_id)

            # Draw bounding box
            cv2.rectangle(
                tracking_frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )
            cv2.putText(
                tracking_frame,
                f"Player {track_id}",
                (int(x1), int(y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Foot point
            cx = (x1 + x2) / 2
            # foot_px = (cx, y2 - 0.05 * (y2 - y1))
            foot_px = (cx, y2)


            # Pixel → meters
            player_m = homography.warp_point(foot_px)
            

            x_m, y_m = float(player_m[0]), float(player_m[1])

            # Reject people not on court
            MARGIN_X = 3   # meters
            MARGIN_Y = 3   # meters

            if not (
                -MARGIN_X <= x_m <= court_width + MARGIN_X and
                -MARGIN_Y <= y_m <= court_length + MARGIN_Y
            ):
                continue

            players_metric.append({
                "track_id": track_id,
                "player_m": (x_m, y_m)
            })

        # --- Select two tennis players (top & bottom) ---
       # -------------------------
        # Robust tennis player selection
        # -------------------------
        # if len(selected_player_ids) < 2 and len(players_metric) >= 2:
        if frame_count > 30 and len(selected_player_ids) < 2 and len(players_metric) >= 2:

            # Sort by court length (Y axis)
            players_metric.sort(key=lambda p: p["player_m"][1])

            near_player = players_metric[0]          # closest to near baseline
            far_player  = players_metric[-1]         # closest to far baseline

            selected_player_ids = {
                near_player["track_id"],
                far_player["track_id"]
            }


        # --- Draw radar points with motion trail ---
        for p in players_metric:
            if p["track_id"] not in selected_player_ids:
                continue

            # Convert meters → radar pixels FIRST
            rx, ry = meters_to_radar(p["player_m"], Radar_W, Radar_H)

            # Assign consistent color
            color = (
                int((37 * p["track_id"]) % 255),
                int((17 * p["track_id"]) % 255),
                int((29 * p["track_id"]) % 255)
            )

            # Initialize trail storage
            player_trails.setdefault(p["track_id"], [])

            # Append current position
            player_trails[p["track_id"]].append((rx, ry))

            # Keep last N points only
            player_trails[p["track_id"]] = player_trails[p["track_id"]][-5:]

            # Draw trail
            for pt in player_trails[p["track_id"]]:
                cv2.circle(radar_frame, pt, 3, color, -1)

            # Draw current position (bigger dot)
            cv2.circle(radar_frame, (rx, ry), 8, color, -1)

            # print("meters:", p["player_m"], "radar:", (rx, ry))
        # --- Overlay radar ---
        tracking_frame = overlay_radar_transparent(
            tracking_frame, radar_frame, alpha=0.6
        )

        out_tracking.write(tracking_frame)

    cap.release()
    out_tracking.release()


if __name__ == "__main__":
    main()
