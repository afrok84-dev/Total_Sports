import numpy as np

def order_court_points(pts):
    pts = np.array(pts)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    topleft = pts[np.argmin(s)]
    bottomright = pts[np.argmax(s)]
    topright = pts[np.argmin(diff)]
    bottomleft = pts[np.argmax(diff)]

    return np.array([topleft, topright, bottomright, bottomleft], dtype=np.float32)
