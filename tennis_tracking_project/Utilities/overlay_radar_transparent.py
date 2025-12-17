import cv2

def overlay_radar_transparent(frame, radar, alpha=0.6, margin=10):
    fh, fw = frame.shape[:2]
    rh, rw = radar.shape[:2]

    x1 = fw - rw - margin
    y1 = margin
    x2 = x1 + rw
    y2 = y1 + rh

    # Extract region of interest
    roi = frame[y1:y2, x1:x2]

    # Alpha blend
    blended = cv2.addWeighted(roi, 1 - alpha, radar, alpha, 0)

    # Replace region
    frame[y1:y2, x1:x2] = blended

    return frame