import sys
sys.path.append("yolov5")

import torch
import cv2
import numpy as np
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox
from yolov5.utils.torch_utils import select_device


class YOLOv5Detector:
    def __init__(self, model_path):
        self.device = select_device("cpu")
        self.model = DetectMultiBackend(model_path, device=self.device)
        self.model.eval()

    def detect(self, frame, img_size=640):
        # ---------------------------------------
        # 1. Preprocess correct way (LETTERBOX)
        # ---------------------------------------
        img = letterbox(frame, img_size, stride=32, auto=True)[0]
        img = img.transpose(2, 0, 1)  # HWC → CHW
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        img = img.unsqueeze(0)  # add batch dimension

        # ---------------------------------------
        # 2. Inference
        # ---------------------------------------
        pred = self.model(img)

        # ---------------------------------------
        # 3. NMS
        # ---------------------------------------
        pred = non_max_suppression(pred, 0.25, 0.45)

        # ---------------------------------------
        # 4. Decode + rescale boxes
        # ---------------------------------------
        detections = []
        det = pred[0]

        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

            # convert to Python list
            for *xyxy, conf, cls in det.cpu().numpy():
                detections.append({
                    "bbox": xyxy,
                    "conf": float(conf),
                    "cls": int(cls)
                })
        return detections
