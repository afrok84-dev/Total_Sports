# --------------------------Line detections ------------------------
# import cv2
# import numpy as np


# class CourtDetector:
#     def __init__(self):
#         self.court_points = None  # will store 4 corner points

#     def detect_lines(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (5, 5), 0)
#         edges = cv2.Canny(blur, 50, 150)

#         lines = cv2.HoughLinesP(
#             edges,
#             rho=1,
#             theta=np.pi / 180,
#             threshold=200,
#             minLineLength=200,
#             maxLineGap=20
#         )

#         return lines

#     def update(self, frame):
#         lines = self.detect_lines(frame)

#         if lines is None:
#             return None

#         #For now: just draw detected lines for debugging
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         return frame
# --------------------------Line detections ------------------------

# --------------------------longest line detection ------------------------
# import cv2
# import numpy as np


# class CourtDetector:
#     def __init__(self):
#         # Store the two longest lines seen across ALL frames
#         # Format: [(length, (x1, y1, x2, y2)), ...]
#         self.best_lines = []

#     def detect_lines(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (5, 5), 0)
#         edges = cv2.Canny(blur, 50, 150)

#         lines = cv2.HoughLinesP(
#             edges,
#             rho=1,
#             theta=np.pi / 180,
#             threshold=25,
#             minLineLength=200,
#             maxLineGap=20
#         )

#         return lines

#     def line_length(self, line):
#         x1, y1, x2, y2 = line
#         return np.hypot(x2 - x1, y2 - y1)

#     def update_best_lines(self, detected_lines):
#         for line in detected_lines:
#             length = self.line_length(line)

#             if len(self.best_lines) < 2:
#                 self.best_lines.append((length, line))
#             else:
#                 # Find the shortest stored line
#                 self.best_lines.sort(key=lambda x: x[0], reverse=True)
#                 shortest_length = self.best_lines[-1][0]

#                 if length > shortest_length:
#                     self.best_lines[-1] = (length, line)

#         # Always keep sorted
#         self.best_lines.sort(key=lambda x: x[0], reverse=True)

#     def update(self, frame):
#         lines = self.detect_lines(frame)

#         if lines is not None:
#             detected = [line[0] for line in lines]
#             self.update_best_lines(detected)

#         # Draw the two longest lines seen so far
#         for _, (x1, y1, x2, y2) in self.best_lines:
#             cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

#         return frame

#     def get_best_lines(self):
#         """Return only the line endpoints"""
#         return [line for _, line in self.best_lines]

# --------------------------longest Line detections ------------------------

# --------------------------Improved white lines detection ------------------------

import cv2
import numpy as np


class CourtDetector:
    def __init__(self):
        self.best_lines = []  # store two longest lines across frames

    def detect_lines(self, frame):
        # ----------------------------------
        # 1. Convert to HSV
        # ----------------------------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ----------------------------------
        # 2. Threshold for white court lines
        # ----------------------------------
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 40, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Optional: clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        # ----------------------------------
        # 3. Blur + Edge detection
        # ----------------------------------
        blur = cv2.GaussianBlur(white_mask, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # ----------------------------------
        # 4. Hough Line Detection
        # ----------------------------------
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=200,
            maxLineGap=30
        )

        return lines, white_mask

    def line_length(self, line):
        x1, y1, x2, y2 = line
        return np.hypot(x2 - x1, y2 - y1)

    def update_best_lines(self, detected_lines):
        for line in detected_lines:
            length = self.line_length(line)

            if len(self.best_lines) < 2:
                self.best_lines.append((length, line))
            else:
                self.best_lines.sort(key=lambda x: x[0], reverse=True)
                if length > self.best_lines[-1][0]:
                    self.best_lines[-1] = (length, line)

        self.best_lines.sort(key=lambda x: x[0], reverse=True)

    def update(self, frame):
        lines, white_mask = self.detect_lines(frame)

        if lines is not None:
            detected = [line[0] for line in lines]
            self.update_best_lines(detected)

        # Draw the two longest lines so far
        for _, (x1, y1, x2, y2) in self.best_lines:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        return frame, white_mask

# --------------------------Improved white lines detection ------------------------