import numpy as np
import cv2


class CourtCornersService:
    def __init__(
        self,
        tennis_field_points: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        image_width: int,
        image_height: int,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.tennis_field_points = tennis_field_points

    def get_field_corners(self) -> np.ndarray:
        """
        Method to compute the four corners of the tennis court from the given field points.
        Returns:
            np.ndarray: Array of shape (4, 2) representing the corners in the order: TL, TR, BR, BL.
        """
        points = self._prepare_points(self.tennis_field_points)
        mask = self._build_mask_from_points(points)
        clean_mask = self._keep_largest_component(mask)
        clean_mask = self._denoise_mask(clean_mask)

        clean_points = self._mask_to_points(clean_mask)
        hull = self._convex_hull(clean_points)
        pts = self._approximate_hull(hull, epsilon=10.0)

        pts = self._ensure_four_corners(pts)
        corners = self._order_corners_tl_tr_br_bl(pts)
        return corners

    def _prepare_points(self, points: np.ndarray) -> np.ndarray:
        """Round to int and drop points outside image bounds."""
        pts = np.round(points).astype(int)
        return self._clip_points_to_image(pts)

    def _clip_points_to_image(self, pts: np.ndarray) -> np.ndarray:
        """Keep only points inside [0..width-1] x [0..height-1]."""
        in_bounds = (
            (pts[:, 0] >= 0) & (pts[:, 0] < self.image_width) &
            (pts[:, 1] >= 0) & (pts[:, 1] < self.image_height)
        )
        return pts[in_bounds]

    def _build_mask_from_points(self, pts: np.ndarray) -> np.ndarray:
        """Binary mask with 1s at (x,y) point locations."""
        mask = np.zeros((self.image_height, self.image_width), np.uint8)
        if pts.size == 0:
            return mask
        mask[pts[:, 1], pts[:, 0]] = 1  # row=y, col=x
        return mask

    def _keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected foreground component."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            raise RuntimeError("No foreground components found in field mask")

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (labels == largest_label).astype(np.uint8)

    def _denoise_mask(self, mask: np.ndarray, ksize: int = 3) -> np.ndarray:
        """Remove small noise/thin protrusions via morphological opening."""
        kernel = np.ones((ksize, ksize), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def _mask_to_points(self, mask: np.ndarray) -> np.ndarray:
        """Convert binary mask to Nx2 float points [[x,y], ...]."""
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            raise RuntimeError("Mask is empty after cleanup")
        return np.stack([xs, ys], axis=1).astype(np.float32)

    def _convex_hull(self, pts_xy: np.ndarray) -> np.ndarray:
        """Compute convex hull for point cloud."""
        field_cv = pts_xy.reshape(-1, 1, 2)
        return cv2.convexHull(field_cv)

    def _approximate_hull(self, hull: np.ndarray, epsilon: float) -> np.ndarray:
        """Approximate hull polygon and return Nx2 points."""
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) < 4:
            raise RuntimeError(f"Got only {len(approx)} points, cannot form quadrilateral")
        return approx.reshape(-1, 2)

    def _ensure_four_corners(self, pts: np.ndarray) -> np.ndarray:
        """If more than 4 points, keep the 4 farthest from centroid."""
        if len(pts) == 4:
            return pts
        if len(pts) < 4:
            raise RuntimeError(f"Need 4 points, got {len(pts)}")

        center = pts.mean(axis=0, keepdims=True)
        d2 = ((pts - center) ** 2).sum(axis=1)
        corner_idx = np.argsort(d2)[-4:]
        return pts[corner_idx]

    def _order_corners_tl_tr_br_bl(self, pts: np.ndarray) -> np.ndarray:
        """Return corners ordered as [TL, TR, BR, BL]."""
        # sort by y (top to bottom), then by x within top/bottom halves
        idx = np.argsort(pts[:, 1])
        top = pts[idx[:2]]
        bottom = pts[idx[2:]]

        top = top[np.argsort(top[:, 0])]
        bottom = bottom[np.argsort(bottom[:, 0])]

        tl, tr = top
        bl, br = bottom
        return np.array([tl, tr, br, bl], dtype=np.float32)
