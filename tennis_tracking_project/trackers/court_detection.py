import cv2
import numpy as np


class CourtDetector:
    def __init__(self):
        self.court_points = None  # will store 4 corner points

    def detect_lines(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=150,
            minLineLength=200,
            maxLineGap=20
        )

        return lines

    def update(self, frame):
        lines = self.detect_lines(frame)

        if lines is None:
            return None

        # For now: just draw detected lines for debugging
        # for line in lines:
        #     x1, y1, x2, y2 = line[0]
        #     cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame
