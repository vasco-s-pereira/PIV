#!/usr/bin/env python3
"""
Build the reference pair part1 expects from a rally folder:

    <refdir>/templateimg.jpg
    <refdir>/court_base_templateimg.mat   {"img_pts": (14, 2)}

img_pts is in the same order as ref_tennis/courtmodel.mat (TennisCourtDetector's
14 keypoints). Missed landmarks are NaN.

Usage:
    uv run python extract_reference.py datasets/rally_04 ref_tennis
"""
import os
import shutil
import sys

import cv2
import numpy as np
import scipy.io as sio
import torch

# TennisCourtDetector still uses np.Inf, removed in NumPy 2.
if not hasattr(np, "Inf"):
    np.Inf = np.inf

from main1 import list_images

DETECTOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TennisCourtDetector")
MODEL_PATH = os.path.join(DETECTOR_DIR, "models", "model.pt")
COURTMODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_tennis", "courtmodel.mat")
MODEL_W, MODEL_H = 1280, 720


def load_model(model_path):
    sys.path.insert(0, DETECTOR_DIR)
    from tracknet import BallTrackerNet

    model = BallTrackerNet(out_channels=15)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device


def _points_to_array(points):
    img_pts = np.full((14, 2), np.nan, dtype=np.float64)
    for i, (x, y) in enumerate(points):
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
            img_pts[i] = (float(x), float(y))
    return img_pts


def detect_keypoints(image, model, device, refine=True):
    """
    Detect the 14 court keypoints in `image`.

    Returns (img_pts_raw, img_pts_snapped), each (14, 2) float64 with NaN for
    missed landmarks:
        img_pts_raw     : the detector's own heatmap peaks (refined sub-pixel).
                          These are the points part1's anchor RANSAC fits, so a
                          single bad keypoint can be rejected.
        img_pts_snapped : the raw points mapped onto the detector's internal
                          court homography (get_trans_matrix). Smoother, but the
                          fit is already baked in; kept only for comparison and
                          to seed the white-line proximity mask.
    """
    sys.path.insert(0, DETECTOR_DIR)
    from homography import get_trans_matrix, refer_kps
    from postprocess import postprocess, refine_kps

    height, width = image.shape[:2]
    resized = cv2.resize(image, (640, 360))
    inp = torch.tensor(np.rollaxis(resized.astype(np.float32) / 255.0, 2, 0)).unsqueeze(0)
    with torch.no_grad():
        out = model(inp.float().to(device))[0]
    pred = torch.sigmoid(out).detach().cpu().numpy()

    sx, sy = width / MODEL_W, height / MODEL_H
    points = []
    for kps_num in range(14):
        heatmap = (pred[kps_num] * 255).astype(np.uint8)
        x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
        if x_pred is None or y_pred is None:
            points.append((None, None))
            continue
        x_pred, y_pred = x_pred * sx, y_pred * sy
        if refine and kps_num not in (8, 12, 9):
            x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
        points.append((x_pred, y_pred))

    snapped = points
    matrix = get_trans_matrix(points)
    if matrix is not None:
        warped = cv2.perspectiveTransform(refer_kps, matrix)
        snapped = [tuple(np.squeeze(p)) for p in warped]

    return _points_to_array(points), _points_to_array(snapped)


def extract_line_pixels(image, img_pts_snapped, radius_px=12):
    """
    Pixel coordinates (M, 2) of white court-line pixels in `image`, kept only
    where they fall within `radius_px` of the painted lines projected through
    the snapped landmarks. Returns an empty (0, 2) array if no anchor can be
    built (too few landmarks).
    """
    from part1 import dlt_homography
    from court_lines import (load_court_model, white_line_mask,
                             project_lines, polyline_proximity_mask)

    model = load_court_model(COURTMODEL)
    ok = np.isfinite(img_pts_snapped).all(axis=1)
    if ok.sum() < 4:
        return np.zeros((0, 2), dtype=np.float64)

    H_img_to_court = dlt_homography(img_pts_snapped[ok], model["pts"][ok])
    if H_img_to_court is None:
        return np.zeros((0, 2), dtype=np.float64)
    H_court_to_img = np.linalg.inv(H_img_to_court)

    h, w = image.shape[:2]
    polylines = project_lines(H_court_to_img, model["lines"], (w, h))
    near = polyline_proximity_mask(polylines, (w, h), radius_px=radius_px)
    white = white_line_mask(image)
    ys, xs = np.nonzero(white & near)
    return np.column_stack([xs, ys]).astype(np.float64)


def ensure_courtmodel(refdir):
    dest = os.path.join(refdir, "courtmodel.mat")
    if os.path.exists(dest) or not os.path.isfile(COURTMODEL):
        return
    os.symlink(os.path.relpath(COURTMODEL, refdir), dest)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    images_dir, refdir = sys.argv[1], sys.argv[2]
    if not os.path.isfile(MODEL_PATH):
        raise SystemExit(f"Missing weights: {MODEL_PATH}")

    images = list_images(images_dir)
    if not images:
        raise SystemExit(f"No images in {images_dir}")
    first = images[0]

    os.makedirs(refdir, exist_ok=True)
    template = os.path.join(refdir, "templateimg.jpg")
    shutil.copy2(first, template)
    ensure_courtmodel(refdir)

    image = cv2.imread(template)
    if image is None:
        raise SystemExit(f"Could not read {template}")
    model, device = load_model(MODEL_PATH)
    img_pts, img_pts_snapped = detect_keypoints(image, model, device)
    line_px = extract_line_pixels(image, img_pts_snapped)

    h, w = image.shape[:2]
    mat_path = os.path.join(refdir, "court_base_templateimg.mat")
    # img_pts holds the *raw* detector peaks (what part1's anchor RANSAC fits).
    sio.savemat(mat_path, {
        "img_pts": img_pts,
        "img_pts_snapped": img_pts_snapped,
        "line_px": line_px,
        "size": np.array([w, h], dtype=np.float64),
    })

    preview = image.copy()
    for x, y in line_px.astype(int):
        preview[y, x] = (0, 255, 0)
    for x, y in img_pts:
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(preview, (int(x), int(y)), 0, (0, 0, 255), 10)
    preview_path = os.path.join(refdir, "templateimg_kps.jpg")
    cv2.imwrite(preview_path, preview)

    found = int(np.isfinite(img_pts).all(axis=1).sum())
    print(f"[extract_reference] {os.path.basename(first)} -> {template}")
    print(f"[extract_reference] {found}/14 keypoints, {len(line_px)} line pixels -> {mat_path}")
    print(f"[extract_reference] preview -> {preview_path}")


if __name__ == "__main__":
    main()
