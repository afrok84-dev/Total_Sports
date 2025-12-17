import cv2
import numpy as np

def extract_points(txt_path):
    """
    txt_path: path to .txt file containing 'x y' per line
    image_shape: (height, width)
    """

    # -----------------------------
    # Step 1: Load pixel coordinates
    # -----------------------------
    points = []
    image_shape=(720, 1280)
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()

            # Remove brackets, parentheses
            line = line.replace("[", "").replace("]", "")
            line = line.replace("(", "").replace(")", "")

            # Replace commas with spaces
            line = line.replace(",", " ")

            # Split and convert
            parts = line.split()
            if len(parts) != 2:
                continue  # skip malformed lines

            x, y = map(int, parts)
            points.append([x, y])

    points = np.array(points, dtype=np.int32)

    # -----------------------------
    # Step 2: Create binary mask
    # -----------------------------
    mask = np.zeros(image_shape, dtype=np.uint8)
    mask[points[:, 1], points[:, 0]] = 255  # y, x indexing

    # -----------------------------
    # Step 3: Find contours
    # -----------------------------
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No contours found")

    # Largest contour = court
    court_contour = max(contours, key=cv2.contourArea)

    # -----------------------------
    # Step 4: Convex hull
    # -----------------------------
    hull = cv2.convexHull(court_contour)

    # -----------------------------
    # Step 5: Extract 4 corners
    # -----------------------------
    hull = hull.reshape(-1, 2)

    # Sum and difference of points
    s = hull.sum(axis=1)
    diff = np.diff(hull, axis=1).reshape(-1)

    top_left     = hull[np.argmin(s)]
    bottom_right = hull[np.argmax(s)]
    top_right    = hull[np.argmin(diff)]
    bottom_left  = hull[np.argmax(diff)]

    corners = np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ], dtype=np.float32)

    return corners 

# Debugging run section and with visualization of mask, 
#NB: Add mask as return of function to visualize if needed

# txt_path = "./mask_coordinates.txt"
# image_shape = (720, 1280)  # height, width
# court_corners = []
# court_corners,mask = extract_court_corners(txt_path, image_shape)

# print('detected corners')
# for corner in court_corners:
#     print(corner)

# vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# for i, (x, y) in enumerate(court_corners):
#     cv2.circle(vis, (x, y), 8, (0, 0, 255), -1)
#     cv2.putText(vis, str(i), (x+10, y),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

# cv2.imwrite("court_corners_debug.png", vis)
# print("Saved visualization to court_corners_debug.png")

