from deep_sort_realtime.deepsort_tracker import DeepSort

class PlayerTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=5)

    def update(self, detections, frame, frame_idx):
        track_inputs = []

        for d in detections:
            print(f"{frame_idx}: DETECTION :{d}")
            # xyxy → xywh
            x1, y1, x2, y2 = [float(v) for v in d["bbox"]]
            w, h = x2 - x1, y2 - y1

            # DeepSORT expected format:
            track_inputs.append([
                [x1, y1, w, h],     # <-- BOX AS NESTED LIST
                float(d["conf"]),   # <-- CONFIDENCE
                int(d["cls"])       # <-- CLASS ID
            ])

        return self.tracker.update_tracks(track_inputs, frame=frame)
