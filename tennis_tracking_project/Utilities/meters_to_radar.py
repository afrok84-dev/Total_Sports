import numpy as np

COURT_W = 8.23   # meters
COURT_L = 23.77  # meters

def meters_to_radar(pt_m, radar_w, radar_h):
    x_m, y_m = float(pt_m[0]), float(pt_m[1])

    # Auto-fix swapped axes
    if x_m > COURT_W and y_m < COURT_L:
        x_m, y_m = y_m, x_m

    # Clamp
    x_m = np.clip(x_m, 0.0, COURT_W)
    y_m = np.clip(y_m, 0.0, COURT_L)

    # Meters → pixels
    x_px = int((x_m / COURT_W) * radar_w)
    y_px = int((y_m / COURT_L) * radar_h)

    # Convert math coords → image coords
    y_px = radar_h - y_px

    return x_px, y_px
