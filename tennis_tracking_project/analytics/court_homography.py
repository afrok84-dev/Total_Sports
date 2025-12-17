
import cv2
import numpy as np

class Homography:
    def __init__(self):
        self.H = None

    def compute(self, src_pts, dst_pts):
        self.H, _ = cv2.findHomography(
            np.float32(src_pts),
            np.float32(dst_pts)
        )
        return self.H

    def warp_point(self, pt):
        if self.H is None:
            raise ValueError("Homography not computed")
        pt = np.array([[pt]], dtype='float32')
        warped = cv2.perspectiveTransform(pt, self.H)
        return warped[0][0]
