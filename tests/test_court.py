#!/usr/bin/env python3
"""
test_court.py - unit tests for the court anchor (part1.py section 6).

Run:  uv run python tests/test_court.py

These use only numpy/scipy so they exercise part1 exactly as the grader does.
A synthetic court->image homography is created, the 14 ITF landmarks are
projected through it with pixel noise, and one landmark is corrupted; the tests
check that the exhaustive RANSAC rejects it, that the fit recovers the true H,
that the white-line refinement reduces the line distance, and that part1 never
imports cv2.
"""
import os
import sys

import numpy as np
import scipy.io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from part1 import (dlt_homography, apply_homography, _anchor_ransac,          # noqa: E402
                   _refine_anchor_landmarks, _refine_anchor_lines,
                   _sample_segments, _median_line_distance, NET_LINE_IDX)
from scipy.spatial import cKDTree                                             # noqa: E402

COURTMODEL = os.path.join(ROOT, "ref_tennis", "courtmodel.mat")


def _load_model():
    m = sio.loadmat(COURTMODEL)
    pts = np.asarray(m["pts"], dtype=np.float64)
    lines = np.asarray(m["lines"], dtype=np.float64).reshape(-1, 2, 2)
    lines = np.delete(lines, NET_LINE_IDX, axis=0)
    return pts, lines


def _synthetic_H(court_pts):
    """A plausible court-metres -> image-pixels homography (broadcast-like)."""
    poly = court_pts[[0, 1, 3, 2]]                         # TL, TR, BR, BL (metres)
    img_quad = np.array([[322.0, 242.0], [942.0, 242.0],
                         [1211.0, 655.0], [72.0, 654.0]])  # a real rally_01 quad
    return dlt_homography(poly, img_quad)


def test_ransac_rejects_corrupt_landmark():
    court_pts, _ = _load_model()
    H = _synthetic_H(court_pts)
    rng = np.random.default_rng(0)
    img = apply_homography(H, court_pts) + rng.normal(0, 0.3, (len(court_pts), 2))
    bad = 7
    img[bad] += np.array([35.0, -28.0])                    # corrupt one landmark

    mask = _anchor_ransac(court_pts, img, threshold=4.0)
    assert not mask[bad], "RANSAC kept the corrupted landmark as an inlier"
    assert mask.sum() >= len(court_pts) - 2, f"too many landmarks dropped: {mask.sum()}"

    Hf = _refine_anchor_landmarks(dlt_homography(court_pts[mask], img[mask]),
                                  court_pts[mask], img[mask])
    # the clean landmarks must reproject within half a pixel of their true image pos
    clean = np.ones(len(court_pts), bool); clean[bad] = False
    err = np.linalg.norm(apply_homography(Hf, court_pts[clean]) - apply_homography(H, court_pts[clean]), axis=1)
    assert np.median(err) < 0.5, f"recovered H off by {np.median(err):.2f}px (median)"
    print(f"  ransac: dropped landmark {bad}, median recovery error {np.median(err):.3f}px  OK")


def test_line_refinement_improves():
    court_pts, lines = _load_model()
    H = _synthetic_H(court_pts)
    size = (1280, 720)

    samples_m = _sample_segments(lines)
    line_px = apply_homography(H, samples_m)               # perfect "white pixels"
    inb = (np.isfinite(line_px).all(1)
           & (line_px[:, 0] >= 0) & (line_px[:, 0] < size[0])
           & (line_px[:, 1] >= 0) & (line_px[:, 1] < size[1]))
    line_px = line_px[inb]

    img_pts = apply_homography(H, court_pts)               # clean landmarks
    # start from a perturbed homography
    Hpert = H.copy()
    Hpert[0, 2] += 6.0; Hpert[1, 2] -= 5.0
    Hpert = Hpert / Hpert[2, 2]

    tree = cKDTree(line_px)
    before = _median_line_distance(Hpert, samples_m, tree, size)
    Href, _ = _refine_anchor_lines(Hpert, court_pts, img_pts, samples_m, line_px, size)
    after = _median_line_distance(Href, samples_m, tree, size)
    assert after <= before, f"line refinement made it worse: {before:.2f} -> {after:.2f}px"
    assert after < 1.0, f"line distance still large after refinement: {after:.2f}px"
    print(f"  line refine: median line distance {before:.2f} -> {after:.2f}px  OK")


def test_part1_no_cv2():
    with open(os.path.join(ROOT, "part1.py")) as f:
        src = f.read()
    for bad in ("import cv2", "from cv2", "import PIL", "import open3d"):
        assert bad not in src, f"part1.py must not contain '{bad}'"
    print("  part1.py imports no cv2 / PIL / open3d  OK")


if __name__ == "__main__":
    test_ransac_rejects_corrupt_landmark()
    test_line_refinement_improves()
    test_part1_no_cv2()
    print("tests/test_court.py: all tests passed")
