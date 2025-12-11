import cv2

from detectors.yolov5_detector import YOLOv5Detector
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker
from analytics.court_homography import CourtHomography
from analytics.ball_physics import BallPhysics


VIDEO_PATH = "input/input_video.mp4"      # <--- your tennis video here
MODEL_PATH = "models/tennis_yolov5.pt"



def main():
    # --- Initialize modules ---
    detector = YOLOv5Detector(MODEL_PATH)
    player_tracker = PlayerTracker()
    ball_tracker = BallTracker()
    homography = CourtHomography()
    physics = BallPhysics()

    cap = cv2.VideoCapture(VIDEO_PATH)

    trajectory = []   # store ball positions for physics

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, 30, 
                      (int(cap.get(3)), int(cap.get(4))))



    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ----------------------------------
        # 1. YOLOv5 Detection
        # ----------------------------------
        detections = detector.detect(frame)

        player_dets = [d for d in detections if d["cls"] == 0]        # assuming class 0 = player
        ball_dets   = [d for d in detections if d["cls"] == 1]        # assuming class 1 = ball

        # ----------------------------------
        # 2. Player Tracking (DeepSORT)
        # ----------------------------------
        player_tracks = player_tracker.update(player_dets, frame)

        # ----------------------------------
        # 3. Ball Tracking + Kalman smoothing
        # ----------------------------------
        if len(ball_dets) > 0:
            # Pick highest-confidence ball det
            ball = max(ball_dets, key=lambda x: x["conf"])
            x1, y1, x2, y2 = ball["bbox"]
            bx, by = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Kalman smoothing
            smoothed = ball_tracker.update(bx, by)
            trajectory.append(smoothed)

            # Draw ball
            cv2.circle(frame, (int(smoothed[0]), int(smoothed[1])), 6, (0, 255, 255), -1)

        # ----------------------------------
        # 4. Draw player tracks
        # ----------------------------------
        for track in player_tracks:
            if not track.is_confirmed():
                continue
            
            x1, y1, x2, y2 = track.to_ltrb()
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Player {track_id}", (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # ----------------------------------
        # 5. Ball Trajectory Visualization
        # ----------------------------------
        for i in range(1, len(trajectory)):
            x1, y1 = trajectory[i-1]
            x2, y2 = trajectory[i]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)

        # ----------------------------------
        # 6. Display frame
        # ----------------------------------
                # cv2.imshow("Tennis Tracking", frame)
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break

        # Write frame to output video
        out.write(frame)
    
    cap.release()
    #cv2.destroyAllWindows()
    out.release()


if __name__ == "__main__":
    main()
