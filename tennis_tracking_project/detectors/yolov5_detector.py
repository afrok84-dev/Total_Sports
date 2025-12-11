
import torch

class YOLOv5Detector:
    def __init__(self, model_path):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    def detect(self, frame):
        results = self.model(frame)
        detections = []
        for *xyxy, conf, cls in results.xyxy[0]:
            detections.append({
                "bbox": xyxy,
                "conf": float(conf),
                "cls": int(cls)
            })
        return detections
