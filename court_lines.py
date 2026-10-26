"""
court_lines.py - shared court-model helpers for the reference extraction, the
court metrics and the overlay video.

cv2 is allowed here (image I/O and morphology). This module is *never* imported
by part1.py, which must stay numpy/scipy-only (tests/test_court.py checks it).

Conventions
-----------
* The court model lives in metres on the court plane (origin at the net centre).
* "court -> image" homographies map metres to pixels (this is inv(Hc)).
* The 10 lines in courtmodel.mat include the net (index 9, y = 0). The net is
  not painted on the ground, so load_court_model drops it by default.
"""
import numpy as np
import scipy.io as sio

try:
    import cv2
except ImportError:                     # keeps the module importable without cv2
    cv2 = None


# Landmark order in courtmodel.mat (TennisCourtDetector's 14 keypoints):
#  0 TL_doubles   1 TR_doubles   2 BL_doubles   3 BR_doubles
#  4 TL_singles   5 BL_singles   6 TR_singles   7 BR_singles
#  8 L_service_top 9 R_service_top 10 L_service_bottom 11 R_service_bottom
# 12 T_service_top 13 T_service_bottom
#
# Each known ITF distance is a pair of landmark indices and its value in metres.
ITF_DISTANCES = {
    "doubles_width":            (0, 1, 10.97),
    "singles_width":            (4, 6, 8.23),
    "service_box_width":        (8, 12, 4.115),
    "baseline_separation":      (0, 2, 23.77),
    "service_line_separation":  (8, 10, 12.80),
    "service_to_baseline":      (4, 8, 5.485),
}

# Doubles court outline, clockwise: TL, TR, BR, BL (landmark indices).
DOUBLES_POLYGON_IDX = (0, 1, 3, 2)

NET_LINE_IDX = 9          # lines[9] is the net (y = 0); not painted on the ground


def load_court_model(path, drop_net=True):
    """
    Read courtmodel.mat.

    Returns a dict:
        pts    (14, 2) float64  landmark coordinates in metres
        names  list[str]        landmark labels
        lines  (L, 2, 2)        painted line segments in metres (net removed)
        polygon (4, 2)          doubles-court outline (TL, TR, BR, BL), metres
    """
    m = sio.loadmat(path)
    pts = np.asarray(m["pts"], dtype=np.float64)
    if pts.shape[0] == 2 and pts.shape[1] != 2:
        pts = pts.T

    names = []
    if "names" in m:
        for n in np.asarray(m["names"]).ravel():
            names.append(str(np.asarray(n).ravel()[0]) if np.asarray(n).size else "")

    lines = np.asarray(m["lines"], dtype=np.float64).reshape(-1, 2, 2)
    if drop_net and len(lines) > NET_LINE_IDX:
        lines = np.delete(lines, NET_LINE_IDX, axis=0)

    polygon = pts[list(DOUBLES_POLYGON_IDX)]
    return {"pts": pts, "names": names, "lines": lines, "polygon": polygon}


def project_points(H, pts):
    """Map (N, 2) points with the 3x3 H. Points behind the camera (w <= 0) or at
    infinity become NaN."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ph = np.column_stack([pts, np.ones(len(pts))]) @ np.asarray(H, float).T
    w = ph[:, 2]
    out = np.full((len(pts), 2), np.nan)
    ok = w > 1e-9
    out[ok] = ph[ok, :2] / w[ok, None]
    return out


def sample_lines(lines, step_m=0.05):
    """For each (2, 2) segment return an (n, 2) array of points in metres, spaced
    ~step_m apart. Returned as a list, one array per line."""
    out = []
    for seg in np.asarray(lines, dtype=np.float64).reshape(-1, 2, 2):
        p0, p1 = seg[0], seg[1]
        length = float(np.linalg.norm(p1 - p0))
        n = max(2, int(np.ceil(length / max(step_m, 1e-6))) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        out.append(p0[None, :] + t * (p1 - p0)[None, :])
    return out


def project_lines(H_court_to_img, lines, size, step_m=0.05):
    """
    Project each painted line into the image.

    Returns a list, one entry per line, of dicts:
        pts   (n, 2) float  pixel coordinates (NaN where w <= 0)
        valid (n,)   bool    finite and inside the [0, w) x [0, h) frame
    """
    w, h = float(size[0]), float(size[1])
    result = []
    for pts_m in sample_lines(lines, step_m):
        xy = project_points(H_court_to_img, pts_m)
        valid = (np.isfinite(xy).all(axis=1)
                 & (xy[:, 0] >= 0) & (xy[:, 0] < w)
                 & (xy[:, 1] >= 0) & (xy[:, 1] < h))
        result.append({"pts": xy, "valid": valid})
    return result


def white_line_mask(img, overlay_mask=None):
    """
    Boolean mask of white court-line pixels.

    White top-hat on the grayscale image isolates thin bright structures (the
    painted lines) regardless of the surface colour; a low-saturation /
    high-value test in HSV rejects bright coloured regions (clay glare, kit).
    `overlay_mask` (if given) marks broadcast graphics to exclude.
    """
    if cv2 is None:
        raise RuntimeError("white_line_mask needs cv2")
    img = np.asarray(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]

    mask = (tophat > 20) & (sat < 70) & (val > 120)
    if overlay_mask is not None:
        mask &= ~np.asarray(overlay_mask, dtype=bool)
    return mask


def polyline_proximity_mask(polylines, size, radius_px=12):
    """
    Binary mask of pixels within `radius_px` of any projected polyline.

    `polylines` is the list returned by project_lines. Lines are drawn on a
    blank canvas and dilated; used to keep only line-adjacent white pixels.
    """
    if cv2 is None:
        raise RuntimeError("polyline_proximity_mask needs cv2")
    w, h = int(round(size[0])), int(round(size[1]))
    canvas = np.zeros((h, w), dtype=np.uint8)
    for pl in polylines:
        xy, valid = pl["pts"], pl["valid"]
        # draw contiguous valid runs as segments
        run = []
        for i in range(len(xy)):
            if valid[i]:
                run.append(xy[i])
            elif len(run) >= 2:
                cv2.polylines(canvas, [np.round(np.array(run)).astype(np.int32)],
                              False, 255, 1)
                run = []
            else:
                run = []
        if len(run) >= 2:
            cv2.polylines(canvas, [np.round(np.array(run)).astype(np.int32)],
                          False, 255, 1)
    if radius_px > 0:
        k = 2 * int(radius_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        canvas = cv2.dilate(canvas, kernel)
    return canvas > 0


def fit_line_tls(points):
    """
    Total-least-squares line fit to (n, 2) points.
    Returns (a, b, c) with a*x + b*y + c = 0 and a^2 + b^2 = 1, or None.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return None
    centroid = pts.mean(axis=0)
    u, s, vt = np.linalg.svd(pts - centroid)
    direction = vt[0]                       # largest singular vector = line dir
    normal = np.array([-direction[1], direction[0]])
    c = -normal @ centroid
    return float(normal[0]), float(normal[1]), float(c)


def intersect_lines(l1, l2):
    """Intersection (x, y) of two lines (a, b, c), or None if (near-)parallel."""
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    x = (-c1 * b2 + c2 * b1) / det
    y = (-a1 * c2 + a2 * c1) / det
    return np.array([x, y])
