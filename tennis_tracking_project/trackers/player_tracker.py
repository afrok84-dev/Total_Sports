
from deep_sort_realtime.deepsort_tracker import DeepSort

class PlayerTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=5)

    def update(self, detections, frame):
        track_inputs = []
        for d in detections:
            x1,y1,x2,y2 = d["bbox"]
            w,h = x2-x1, y2-y1
            track_inputs.append([x1, y1, w, h, d["conf"], d["cls"]])
        return self.tracker.update_tracks(track_inputs, frame=frame)
