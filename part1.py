"""
part1.py - Planar tracking and homography estimation (PIV Project, Part 1).

Only numpy / scipy are used (no cv2). DLT and RANSAC are written out by hand.

Outputs, one file per frame:  homography_NNNN.mat  ->  {"H": H, "Hc": Hc}
    H  : frame NNNN -> reference image (templateimg.jpg), pixels
    Hc : frame NNNN -> court model, metres      (Hc = H_ref_to_court @ H)
plus tracking_log.json (method used per frame, so cuts can be checked later).

Feature matching and file I/O (section 5) [membro 2]:
    * RootSIFT descriptors, ratio test + mutual check by exact brute force
    * scoreboard / channel-logo filter: keypoints that stay put while the camera moves
    * two-pass matching: global -> H -> guided matching around that same H

Tracking strategy (hybrid with re-anchoring):
    * sequential : match frame i to the last good frame, compose with its H
                   (small baseline, many matches, but drifts)
    * direct     : match frame i to the reference (no drift, fails when the
                   camera is far from the reference)
    * if both work, the direct estimate re-anchors the track unless it
      disagrees wildly with the sequential one (repetitive-line mismatch)
    * cuts are detected by continuity: frame i matches frame i-1 far worse than usual.
      After a cut the track is "lost": no composition across it, and frames are written
      with NaN H / Hc (method "no_court" in tracking_log.json) until a re-anchor: a strong
      direct estimate, consistent with a pan-tilt-zoom of the reference camera
      (rotation_residual), confirmed by the next frame. Close-ups, crowd shots and other
      cameras (whose matches land on the stands) never pass that test.
    * a frame with no acceptable estimate while tracking is also NaN ("no_estimate")
"""

import os
import re
import glob
import json
import struct
from itertools import combinations

import numpy as np
import scipy.io as sio
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from scipy.ndimage import binary_closing, binary_dilation


# ------------------------------------------------------------------------------
# Tunable parameters
# ------------------------------------------------------------------------------
RATIO_THRESH = 0.8          # Lowe ratio test (on RootSIFT; 0.8 tested on the synthetic set)
RANSAC_THRESH = 3.5         # inlier threshold (px, symmetric transfer error)
MIN_MATCHES = 12            # matches needed before even running RANSAC
MIN_INLIERS_DIRECT = 15     # inliers needed to accept a direct estimate
MIN_INLIERS_SEQ = 12        # inliers needed to accept a sequential estimate
MIN_SPAN = (0.30, 0.20)     # inliers must spread over this fraction of the frame
                            # (rejects scoreboard / logo clusters)
AREA_RANGE = (0.05, 20.0)   # allowed warped-frame area / frame area
GATE_PX = 60.0              # max disagreement direct vs sequential (px in reference)
STRONG_INLIERS = 80         # a direct estimate this strong is trusted regardless
GUIDED_RADIUS = 3.0         # px: window of the 2nd matching pass, around the pair's own H.
                            # Keep it below RANSAC_THRESH: at 4 px, near-miss wrong matches
                            # (3-3.5 px off) got in as inliers and biased 2 frames by ~6 px
OVERLAY_SAMPLES = 40        # frames used to find the scoreboard / logos fixed on screen
CUT_RATIO = 0.25            # cut: inliers to the previous frame below this fraction of the
CUT_WINDOW = 15             #      median over the last CUT_WINDOW tracked frames
CUT_MIN_HISTORY = 5         #      (needs this many tracked frames before it can fire)
REANCHOR_FRAC = 0.4         # re-anchor after a cut: direct inliers >= this x median while tracking
                            # (rallies 01-04: tracking >= 550, close-ups / other cameras <= 265)
ROT_MAX = 0.03              # re-anchor: max rotation-consistency residual (tracking p99 0.014)
CONFIRM_PX = 10.0           # re-anchor: two consecutive candidates must agree within this


# ------------------------------------------------------------------------------
# 1. DLT with Hartley normalisation
# ------------------------------------------------------------------------------
def normalize_points(pts):
    """
    Hartley normalisation: move the centroid to the origin and scale so the
    mean distance to the origin is sqrt(2).

    Returns (pts_norm, T) with  [x_norm, 1]^T = T [x, 1]^T.
    """
    pts = np.asarray(pts, dtype=np.float64)
    mean = pts.mean(axis=0)
    centered = pts - mean
    mean_dist = np.mean(np.sqrt(np.sum(centered ** 2, axis=1)))
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 1e-8 else 1.0

    T = np.array([[scale, 0.0,   -scale * mean[0]],
                  [0.0,   scale, -scale * mean[1]],
                  [0.0,   0.0,    1.0]])
    return centered * scale, T


def normalize_scale(H):
    """Fix the arbitrary scale of H (H[2,2] = 1, or unit norm if that is ~0)."""
    if abs(H[2, 2]) > 1e-10:
        return H / H[2, 2]
    n = np.linalg.norm(H)
    return H / n if n > 1e-10 else H


def dlt_homography(pts_src, pts_dst):
    """
    Direct Linear Transform:  pts_dst ~ H pts_src.

    For one correspondence x=(x,y,1) -> x'=(u,v,1), the relation x' ~ H x means
        u = (h1.x) / (h3.x)      v = (h2.x) / (h3.x)
    where h1, h2, h3 are the rows of H. Cross-multiplying gives two equations
    that are linear in the 9 entries of H:
        -x*h11 - y*h12 - h13                    + u*x*h31 + u*y*h32 + u*h33 = 0
                          -x*h21 - y*h22 - h23  + v*x*h31 + v*y*h32 + v*h33 = 0

    Stacking N correspondences gives a (2N x 9) matrix A with A h = 0.
    H has 8 degrees of freedom (the scale is free), so 4 correspondences give
    8 equations and A has rank 8: its 1-D null space is H.  With more than 4
    points the system is over-determined and the right singular vector of the
    smallest singular value is the least-squares solution (min ||A h||, ||h||=1).

    Returns the 3x3 matrix, or None if N < 4 or the SVD fails.
    """
    pts_src = np.asarray(pts_src, dtype=np.float64)
    pts_dst = np.asarray(pts_dst, dtype=np.float64)
    N = len(pts_src)
    if N < 4:
        return None

    # 1. normalise both point sets (conditioning)
    src_n, T_src = normalize_points(pts_src)
    dst_n, T_dst = normalize_points(pts_dst)

    # 2. build A (2N x 9)
    A = np.zeros((2 * N, 9))
    for i in range(N):
        x, y = src_n[i]
        u, v = dst_n[i]
        A[2 * i]     = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
        A[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]

    # 3. solve A h = 0.  full_matrices=True (default) matters when N = 4:
    #    A is 8x9, and the 9th right-singular vector is the null vector.
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    H_norm = Vt[-1].reshape(3, 3)

    # 4. undo the normalisation:  H = T_dst^-1  H_norm  T_src
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    return normalize_scale(H)


# ------------------------------------------------------------------------------
# 2. Projection, errors, validity checks
# ------------------------------------------------------------------------------
def apply_homography(H, pts):
    """Project (N,2) points with H. Points that map to infinity become NaN."""
    pts = np.asarray(pts, dtype=np.float64)
    ph = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    w = ph[:, 2]
    out = np.full((len(pts), 2), np.nan)
    ok = np.abs(w) > 1e-12
    out[ok] = ph[ok, :2] / w[ok, None]
    return out


def symmetric_transfer_error(H, pts_src, pts_dst):
    """
    Per-correspondence error (pixels):
        sqrt( 0.5 * ( |dst - H src|^2 + |src - H^-1 dst|^2 ) )
    Invalid points get +inf.
    """
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(len(pts_src), np.inf)

    fwd = np.sum((apply_homography(H, pts_src) - pts_dst) ** 2, axis=1)
    bwd = np.sum((apply_homography(H_inv, pts_dst) - pts_src) ** 2, axis=1)
    err = np.sqrt(0.5 * (fwd + bwd))
    err[~np.isfinite(err)] = np.inf
    return err


def is_valid_homography(H):
    """Finite, invertible and not absurdly ill-conditioned."""
    return (H is not None and np.all(np.isfinite(H))
            and abs(np.linalg.det(H)) > 1e-12
            and np.linalg.cond(H) < 1e10)


def is_sample_degenerate(src4, dst4, tol=1e-3):
    """
    True if any 3 of the 4 points are (nearly) collinear in the source or the
    destination. The triangle area is measured relative to the squared spread
    of the 4 points, so the test does not depend on units (pixels or metres).
    """
    for pts in (src4, dst4):
        scale = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
        if scale < 1e-9:
            return True
        for i, j, k in combinations(range(4), 3):
            area = 0.5 * abs((pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1])
                             - (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0]))
            if area / scale ** 2 < tol:
                return True
    return False


def is_plausible(H, w, h):
    """
    Sanity check for a frame->reference homography: the frame rectangle must
    land on a convex, non-mirrored quadrilateral of reasonable size.
    """
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    if not is_valid_homography(H):
        return False
    ph = np.column_stack([corners, np.ones(4)]) @ H.T
    if not (np.all(ph[:, 2] > 1e-12) or np.all(ph[:, 2] < -1e-12)):
        return False                        # corners on both sides of the horizon
    q = ph[:, :2] / ph[:, 2:3]

    # convexity: all consecutive edge cross-products have the same sign
    edges = np.roll(q, -1, axis=0) - q
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] \
        - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if not np.all(cross > 0):               # > 0 also rules out mirroring
        return False

    area = 0.5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])
    ratio = area / (w * h)
    return AREA_RANGE[0] <= ratio <= AREA_RANGE[1]


# ------------------------------------------------------------------------------
# 3. Non-linear refinement
# ------------------------------------------------------------------------------
def refine_homography(H0, pts_src, pts_dst):
    """
    Levenberg-Marquardt on the symmetric transfer error. The residual vector
    holds the x and y components of both the forward and backward errors
    (4N values), so the cost is smooth. Used on RANSAC inliers only.
    The result is kept only if the mean error actually decreases.
    """
    if len(pts_src) < 4 or abs(H0[2, 2]) < 1e-10:
        return H0
    H0 = normalize_scale(H0)

    def residuals(p):
        H = np.append(p, 1.0).reshape(3, 3)
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return np.full(4 * len(pts_src), 1e3)
        r = np.concatenate([(apply_homography(H, pts_src) - pts_dst).ravel(),
                            (apply_homography(H_inv, pts_dst) - pts_src).ravel()])
        r[~np.isfinite(r)] = 1e3
        return r

    try:
        res = least_squares(residuals, H0.ravel()[:8], method="lm",
                            x_scale="jac", max_nfev=200)
    except Exception:
        return H0

    H = np.append(res.x, 1.0).reshape(3, 3)
    if not is_valid_homography(H):
        return H0
    before = np.mean(symmetric_transfer_error(H0, pts_src, pts_dst))
    after = np.mean(symmetric_transfer_error(H, pts_src, pts_dst))
    return H if after < before else H0


# ------------------------------------------------------------------------------
# 4. RANSAC
# ------------------------------------------------------------------------------
def ransac_homography(pts_src, pts_dst, max_iters=2000, threshold=3.0,
                      min_inliers=10, confidence=0.995, rng=None):
    """
    RANSAC for a homography.

    Loop:
        1. draw a minimal sample of 4 correspondences
        2. reject it if 3 of the points are collinear
        3. fit H with the DLT
        4. score H: inliers are the points with symmetric transfer error < threshold
        5. keep the model with the most inliers (ties: smaller total error)
        6. shrink the number of iterations needed:
               N = log(1 - confidence) / log(1 - w^4),  w = inlier ratio so far
    Finally H is re-estimated with the DLT on all inliers, refined with LM, and
    the inliers are recomputed with the final H.

    Returns (H, inlier_mask). H is None if fewer than `min_inliers` inliers
    were found.
    """
    pts_src = np.asarray(pts_src, dtype=np.float64)
    pts_dst = np.asarray(pts_dst, dtype=np.float64)
    N = len(pts_src)
    no_inliers = np.zeros(N, dtype=bool)
    if N < max(4, min_inliers):
        return None, no_inliers

    if rng is None:
        rng = np.random.default_rng(42)          # fixed seed: reproducible runs

    best_mask, best_n, best_cost = no_inliers, 0, np.inf
    n_iters = max_iters
    it = 0
    while it < n_iters:
        it += 1

        idx = rng.choice(N, size=4, replace=False)          # 1. sample
        if is_sample_degenerate(pts_src[idx], pts_dst[idx]):  # 2. degeneracy
            continue
        H = dlt_homography(pts_src[idx], pts_dst[idx])        # 3. fit
        if not is_valid_homography(H):
            continue

        err = symmetric_transfer_error(H, pts_src, pts_dst)   # 4. score
        mask = err < threshold
        n_in = int(mask.sum())
        cost = float(err[mask].sum())

        if n_in > best_n or (n_in == best_n and cost < best_cost):   # 5. keep best
            best_mask, best_n, best_cost = mask, n_in, cost

            w = n_in / N                                            # 6. adapt
            if w ** 4 >= 1.0:
                n_iters = min(n_iters, it)
            elif w ** 4 > 1e-12:
                needed = np.log(1.0 - confidence) / np.log(1.0 - w ** 4)
                n_iters = min(n_iters, int(np.ceil(needed)))

    if best_n < min_inliers:
        return None, best_mask

    # final least-squares fit on all inliers, then non-linear refinement
    src_in, dst_in = pts_src[best_mask], pts_dst[best_mask]
    H = dlt_homography(src_in, dst_in)
    if not is_valid_homography(H):
        return None, best_mask
    H = refine_homography(H, src_in, dst_in)

    mask = symmetric_transfer_error(H, pts_src, pts_dst) < threshold
    if mask.sum() < min_inliers:
        return None, mask
    return H, mask


# ------------------------------------------------------------------------------
# 5. Features, matching and I/O  [membro 2]
# ------------------------------------------------------------------------------
# Features are tuples (pts (N,2) float64 [x, y], des (N,128) float32), as before.
# Matches go cur -> ref: src = cur pts, dst = ref pts, and the homography is H @ src ~ dst.

IMG_EXTS = (".jpg", ".jpeg", ".png")


def rootsift(des, eps=1e-12):
    """
    RootSIFT (Arandjelovic & Zisserman 2012): L1-normalise, then square root.
    Euclidean distance between RootSIFT vectors is the Hellinger distance between
    the original histograms, which compares SIFT descriptors better than plain L2.
    The result has unit L2 norm, so distances lie in [0, sqrt(2)].
    """
    des = np.asarray(des, dtype=np.float32)
    if len(des) == 0:
        return des
    des = np.maximum(des, 0)
    return np.sqrt(des / (des.sum(axis=1, keepdims=True) + eps))


def load_feature_mat(mat_path):
    """
    Load 'kp' of shape (130, N) from a feature .mat file (pivist/features format):
    row 0 = x, row 1 = y, rows 2..129 = SIFT descriptor.
    Returns pts (N,2) and RootSIFT descriptors (N,128).
    Copes with empty files (frames with no keypoints), a transposed matrix and the
    zero padding the VisionIST box adds when kp[i] is saved whole.
    """
    data = sio.loadmat(mat_path)
    if "kp" not in data:
        raise KeyError(f"{mat_path} has no 'kp' entry")
    kp = np.asarray(data["kp"])
    if kp.size == 0:
        return np.zeros((0, 2)), np.zeros((0, 128), np.float32)
    if kp.shape[0] != 130 and kp.ndim == 2 and kp.shape[1] == 130:
        kp = kp.T
    if kp.shape[0] != 130:
        raise ValueError(f"{mat_path}: expected kp with 130 rows, got {kp.shape}")
    pts = kp[:2, :].T.astype(np.float64)
    des = kp[2:, :].T.astype(np.float32)
    keep = des.any(axis=1)                   # all-zero rows are padding, not keypoints
    return pts[keep], rootsift(des[keep])


def _knn2_brute(qa, db, chunk=2048):
    """
    Two nearest neighbours in db of every row of qa (exact, by matrix product in
    blocks), plus the nearest row of qa for every row of db (for the mutual check).
    In 128-D this is ~7x faster than cKDTree and gives the same answer.
    """
    na, nb = len(qa), len(db)
    idx = np.full((na, 2), -1, dtype=np.int64)
    dist = np.full((na, 2), np.inf, dtype=np.float32)
    col_d = np.full(nb, np.inf, dtype=np.float32)
    col_i = np.full(nb, -1, dtype=np.int64)
    db_sq = np.einsum("ij,ij->i", db, db)
    cols = np.arange(nb)
    for s in range(0, na, chunk):
        q = qa[s:s + chunk]
        d2 = np.einsum("ij,ij->i", q, q)[:, None] + db_sq[None, :] - 2.0 * (q @ db.T)
        np.maximum(d2, 0.0, out=d2)
        rows = np.arange(len(q))
        if nb >= 2:
            part = np.argpartition(d2, 1, axis=1)[:, :2]
            pd = d2[rows[:, None], part]
            order = np.argsort(pd, axis=1)
            idx[s:s + len(q)] = np.take_along_axis(part, order, 1)
            dist[s:s + len(q)] = np.take_along_axis(pd, order, 1)
        else:
            idx[s:s + len(q), 0] = 0
            dist[s:s + len(q), 0] = d2[:, 0]
        cm = d2.argmin(axis=0)
        cd = d2[cm, cols]
        better = cd < col_d
        col_d[better] = cd[better]
        col_i[better] = cm[better] + s
    return idx, np.sqrt(dist), col_i


def _unique_targets(i1, i2, d):
    """Sort by distance and keep at most one match per target index."""
    order = np.argsort(d, kind="stable")
    i1, i2, d = i1[order], i2[order], d[order]
    _, first = np.unique(i2, return_index=True)
    first.sort()
    return i1[first], i2[first], d[first]


def match_descriptors(des1, des2, ratio_thresh=RATIO_THRESH, mutual_check=True):
    """
    Global matching: Lowe's ratio test (d1 < ratio * d2) and, optionally, the mutual
    check (i1 must also be the nearest neighbour of its match). Returns indices
    (i1, i2) and distances.
    """
    empty = np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    if len(des1) == 0 or len(des2) == 0:
        return empty
    idx, dist, back = _knn2_brute(des1, des2)
    i1 = np.arange(len(des1))
    keep = dist[:, 0] < ratio_thresh * dist[:, 1]            # ratio test
    if mutual_check:
        keep &= back[idx[:, 0]] == i1                        # 2->1 must agree
    return _unique_targets(i1[keep], idx[keep, 0], dist[keep, 0])


def match_sift_features(pts1, des1, pts2, des2, ratio_thresh=RATIO_THRESH,
                        mutual_check=True):
    """Same interface as before: returns the matched points of both sets."""
    if len(des1) < 4 or len(des2) < 4:
        return np.empty((0, 2)), np.empty((0, 2))
    i1, i2, _ = match_descriptors(des1, des2, ratio_thresh, mutual_check)
    return pts1[i1], pts2[i2]


def match_guided(pts1, des1, pts2, des2, H_pred, radius=GUIDED_RADIUS,
                 ratio_thresh=RATIO_THRESH, max_candidates=16):
    """
    Guided matching: point i of set 1 may only match points of set 2 that lie within
    `radius` pixels of H_pred(i). Inside that window the ratio test only compares
    plausible candidates, so the repeated court lines (whose second neighbour is
    another identical corner) are no longer thrown away.

    H_pred must be GOOD (error well below `radius`): use the H of the same frame
    pair (see match_two_pass). With a stale prediction, e.g. the previous frame's H
    while the camera moves ~30 px/frame, the window misses the true match and picks
    wrong neighbours that are all pulled towards the prediction: a false consensus
    that RANSAC may accept (25-60 px errors in the synthetic test).
    """
    empty = np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    if len(des1) == 0 or len(des2) == 0 or H_pred is None:
        return empty
    pred = apply_homography(H_pred, pts1)
    valid = np.flatnonzero(np.isfinite(pred).all(axis=1))
    if len(valid) == 0:
        return empty
    k = int(min(max_candidates, len(pts2)))
    _, cand = cKDTree(pts2).query(pred[valid], k=k, distance_upper_bound=radius, workers=-1)
    cand = np.asarray(cand).reshape(len(valid), k)
    missing = cand >= len(pts2)                               # no neighbour inside the radius
    has_any = ~missing[:, 0]
    valid, cand, missing = valid[has_any], cand[has_any], missing[has_any]
    if len(valid) == 0:
        return empty

    a = des1[valid][:, None, :]                               # (n,1,128)
    b = des2[np.where(missing, 0, cand)]                      # (n,k,128)
    d = np.sqrt(np.maximum(((a - b) ** 2).sum(-1), 0)).astype(np.float32)
    d[missing] = np.inf
    order = np.argsort(d, axis=1)
    rows = np.arange(len(valid))
    d1 = d[rows, order[:, 0]]
    d2 = d[rows, order[:, 1]] if k >= 2 else np.full(len(valid), np.inf, np.float32)
    keep = d1 < ratio_thresh * d2                             # one candidate only: d2 = inf
    return _unique_targets(valid[keep], cand[rows[keep], order[keep, 0]], d1[keep])


def match_two_pass(cur, ref, estimate, min_matches=MIN_MATCHES):
    """
    Two-pass matching (guided matching, Hartley & Zisserman):
        1. global matching -> estimate -> H0
        2. guided matching in a GUIDED_RADIUS window around H0 of this same pair
           -> estimate -> H1 (kept only if it has at least as many inliers as H0)
    estimate(src, dst) -> (H or None, inlier_mask).
    Returns (H, inliers, src, dst); H is None if the first pass fails.
    """
    (p1, d1), (p2, d2) = cur, ref
    i1, i2, _ = match_descriptors(d1, d2)
    if len(i1) < min_matches:
        return None, np.zeros(len(i1), bool), p1[i1], p2[i2]
    src, dst = p1[i1], p2[i2]
    H0, inl0 = estimate(src, dst)
    if H0 is None:
        return None, inl0, src, dst

    j1, j2, _ = match_guided(p1, d1, p2, d2, H0)
    if len(j1) >= min_matches:
        H1, inl1 = estimate(p1[j1], p2[j2])
        if H1 is not None and inl1.sum() >= inl0.sum():
            return H1, inl1, p1[j1], p2[j2]
    return H0, inl0, src, dst


def find_static_overlay(ref, feature_paths, size=None, n_samples=OVERLAY_SAMPLES,
                        still_px=1.5, moved_px=4.0, min_moved_frac=0.3,
                        min_hits=3, min_frac=0.6, cell=16):
    """
    Find the image regions that do NOT move with the camera (TV graphics: scoreboard,
    channel logo). Their matches all agree with H = identity, so they can win RANSAC
    when the court has few features, and on a cut to another camera they give a
    "perfect" match (105 false inliers per cut frame in the synthetic test).

    On ~n_samples frames spread over the sequence, match globally to the reference.
    In a frame where the camera moved (at least min_moved_frac of the matches moved
    more than moved_px), a reference keypoint that stayed put (< still_px) scores a
    hit. It is an overlay if it has >= min_hits hits and stayed put in >= min_frac
    of the moving-camera frames where it was matched. If the camera never moves,
    nothing is marked (and then the identity is right anyway).

    Returns a grid mask {"cell", "grid", "n_keypoints"} for drop_masked().
    """
    pts, des = ref
    if size is None:
        size = (int(pts[:, 0].max()) + 1, int(pts[:, 1].max()) + 1) if len(pts) else (1, 1)
    w, h = size
    grid = np.zeros((int(np.ceil(h / cell)), int(np.ceil(w / cell))), bool)
    mask = {"cell": cell, "grid": grid, "n_keypoints": 0}
    if len(pts) == 0 or len(feature_paths) == 0:
        return mask

    sample = np.unique(np.linspace(0, len(feature_paths) - 1,
                                   min(n_samples, len(feature_paths))).round().astype(int))
    hits = np.zeros(len(pts), int)
    seen = np.zeros(len(pts), int)
    for k in sample:
        f_pts, f_des = try_load(feature_paths[k])
        if f_des is None:
            continue
        i1, i2, _ = match_descriptors(f_des, des)
        if len(i1) < 8:
            continue
        disp = np.linalg.norm(f_pts[i1] - pts[i2], axis=1)
        if np.mean(disp > moved_px) < min_moved_frac:
            continue                                          # camera still: no information
        seen[i2] += 1
        hits[i2[disp < still_px]] += 1
    static = (hits >= min_hits) & (hits >= min_frac * np.maximum(seen, 1))

    cx = np.clip((pts[static, 0] // cell).astype(int), 0, grid.shape[1] - 1)
    cy = np.clip((pts[static, 1] // cell).astype(int), 0, grid.shape[0] - 1)
    grid[cy, cx] = True
    if grid.any():
        grid = binary_closing(grid, iterations=2)             # merge the scoreboard letters
        grid = binary_dilation(grid, iterations=1)
    mask.update(grid=grid, n_keypoints=int(static.sum()))
    return mask


def drop_masked(feat, mask):
    """Remove the keypoints that fall on marked cells of the overlay mask."""
    pts, des = feat
    if mask is None or des is None or len(pts) == 0 or not mask["grid"].any():
        return feat
    g, c = mask["grid"], mask["cell"]
    cx = np.clip((pts[:, 0] // c).astype(int), 0, g.shape[1] - 1)
    cy = np.clip((pts[:, 1] // c).astype(int), 0, g.shape[0] - 1)
    keep = ~g[cy, cx]
    return pts[keep], des[keep]


def image_size(path):
    """(width, height) of a JPEG or PNG, read from the header only (no cv2 / PIL)."""
    with open(path, "rb") as f:
        head = f.read(26)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", head[16:24])
            return int(w), int(h)
        if head[:2] != b"\xff\xd8":
            raise ValueError(f"{path}: unsupported image format")
        f.seek(2)
        while True:
            byte = f.read(1)
            while byte and byte != b"\xff":
                byte = f.read(1)
            while byte == b"\xff":
                byte = f.read(1)
            if not byte:
                raise ValueError(f"{path}: JPEG without SOF marker")
            marker = byte[0]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                continue
            seg_len = struct.unpack(">H", f.read(2))[0]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                _, h, w = struct.unpack(">BHH", f.read(5))
                return int(w), int(h)
            f.seek(seg_len - 2, 1)


def list_frames(images_dir):
    """[(number, number_string, name_without_ext)] sorted by frame number.
    number_string keeps the leading zeros, so the output NNNN matches the input."""
    frames = []
    for f in os.listdir(images_dir):
        if not f.lower().endswith(IMG_EXTS):
            continue
        name = os.path.splitext(f)[0]
        m = re.search(r"(\d+)$", name)
        if m:
            frames.append((int(m.group(1)), m.group(1), name))
        else:
            print(f"[part1] ignoring {f}: no frame number at the end of the name")
    return sorted(frames)


def try_load(path, mask=None):
    """(pts, des) of a feature file, without overlay keypoints; (None, None) if missing."""
    if not os.path.exists(path):
        return None, None
    try:
        return drop_masked(load_feature_mat(path), mask)
    except Exception as e:
        print(f"[part1] could not read {path}: {e}")
        return None, None


def save_homography(out_dir, num_str, H, Hc):
    """out_dir/homography_NNNN.mat = {"H": (3,3), "Hc": (3,3)}, float64, H[2,2] = 1."""
    H = normalize_scale(np.asarray(H, dtype=np.float64).reshape(3, 3))
    Hc = np.asarray(Hc, dtype=np.float64).reshape(3, 3)
    if np.all(np.isfinite(Hc)):
        Hc = normalize_scale(Hc)
    sio.savemat(os.path.join(out_dir, f"homography_{num_str}.mat"), {"H": H, "Hc": Hc})


def missing_outputs(out_dir, frames):
    """Frames with no homography_NNNN.mat. Must be empty at the end of part1."""
    return [name for _, num_str, name in frames
            if not os.path.isfile(os.path.join(out_dir, f"homography_{num_str}.mat"))]


# ------------------------------------------------------------------------------
# 6. Court anchor
# ------------------------------------------------------------------------------
def _as_points(a):
    a = np.asarray(a, dtype=np.float64)
    return a.T if a.shape[0] == 2 and a.shape[1] != 2 else a


def load_court_anchor(refdir, images_dir=None):
    """
    H_ref_to_court (reference-image pixels -> court metres), by DLT on landmarks.

    Expects in refdir (or, for the landmark file, in the images directory):
        courtmodel.mat    {"pts": (N,2) metres, "names": ...}
        court_base*.mat   {"img_pts" (or "points"): (N,2) pixel positions of the same landmarks
                           in templateimg.jpg}; an optional "court_pts" key
                           overrides the model points (for a subset of landmarks)
    Raises if the anchor cannot be built: a silent identity would make Hc garbage.
    """
    cm_path = os.path.join(refdir, "courtmodel.mat")
    if not os.path.exists(cm_path):
        raise FileNotFoundError(f"{cm_path} not found")
    court_pts = _as_points(sio.loadmat(cm_path)["pts"])

    base_files = []
    for d in (refdir, images_dir):                    # README: either directory
        if d:
            base_files += sorted(glob.glob(os.path.join(d, "court_base*.mat")))
    if not base_files:
        raise FileNotFoundError(
            f"No court_base*.mat in {refdir} or {images_dir}: the anchor needs "
            "the pixel positions of the court landmarks in the reference image.")
    base = sio.loadmat(base_files[0])
    key = next((k for k in ("img_pts", "points") if k in base), None)
    if key is None:
        raise KeyError(f"{base_files[0]} has neither 'img_pts' nor 'points'")
    img_pts = _as_points(base[key])
    if "court_pts" in base:
        court_pts = _as_points(base["court_pts"])

    if img_pts.shape != court_pts.shape:
        raise ValueError(f"Landmark count mismatch: image {img_pts.shape} vs court {court_pts.shape}")
    ok = np.isfinite(img_pts).all(axis=1) & np.isfinite(court_pts).all(axis=1)
    if not ok.all():                                  # detector missed some keypoints
        print(f"[part1] Anchor: dropping {int((~ok).sum())} undetected landmark(s)")
    img_pts, court_pts = img_pts[ok], court_pts[ok]
    if len(img_pts) < 4:
        raise ValueError(f"Need >= 4 valid landmark pairs, got {len(img_pts)}")

    H = dlt_homography(img_pts, court_pts)
    if not is_valid_homography(H):
        raise ValueError("Court anchor DLT produced a degenerate homography")

    rms_cm = 100 * np.sqrt(np.mean(np.sum((apply_homography(H, img_pts) - court_pts) ** 2, axis=1)))
    print(f"[part1] Anchor from {os.path.basename(base_files[0])}: "
          f"{len(img_pts)} landmarks, RMS residual {rms_cm:.1f} cm")
    return H


# ------------------------------------------------------------------------------
# 7. Tracking helpers
# ------------------------------------------------------------------------------
def estimate_pair(cur, ref, extent, min_inliers):
    """
    Homography cur -> ref from two-pass SIFT matching + RANSAC.
    Returns (H, n_inliers), or (None, 0) if matches are too few, RANSAC fails,
    or the inliers are bunched in one place (scoreboard / logo).
    """
    if cur[1] is None or ref[1] is None:
        return None, 0
    estimate = lambda s, d: ransac_homography(s, d, threshold=RANSAC_THRESH,
                                              min_inliers=min_inliers)
    H, inl, src, _ = match_two_pass(cur, ref, estimate)          # [membro 2]
    if H is None:
        return None, 0
    span = np.ptp(src[inl], axis=0) / extent
    if span[0] < MIN_SPAN[0] or span[1] < MIN_SPAN[1]:
        return None, 0
    return H, int(inl.sum())


def is_cut(n_cons, history):
    """A cut (or any break in continuity): the previous frame matches far worse than usual."""
    if len(history) < CUT_MIN_HISTORY:
        return False
    return n_cons < CUT_RATIO * np.median(history[-CUT_WINDOW:])


def rotation_residual(H, w, h):
    """
    How far H is from a homography a pan-tilt-zoom camera can produce, H ~ K_r R K_i^-1
    (same centre, square pixels, principal point at the image centre, unknown focal
    lengths f_r, f_i). With M = K_r^-1 H K_i, M^T M must be a multiple of I; we fit
    f_r, f_i and return the residual. A different camera position gives H not of this
    form, even when it is temporally consistent (matches on the stands of another view).
    """
    c = np.array([w / 2.0, h / 2.0])
    K = lambda f: np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1.0]])

    def res(lf):
        fr, fi = np.exp(lf)
        M = np.linalg.solve(K(fr), H @ K(fi))
        G = M.T @ M
        G = 3.0 * G / np.trace(G)
        return (G - np.eye(3))[np.triu_indices(3)]

    best = min((least_squares(res, np.log([f0, f0]), bounds=(np.log(0.15 * w), np.log(15.0 * w)))
                for f0 in (0.6 * w, 1.2 * w, 2.4 * w)), key=lambda r: r.cost)
    return float(np.sqrt(2.0 * best.cost))


def disagreement(Ha, Hb, w, h):
    """Mean distance (reference pixels) between two homographies on 5 frame points."""
    g = np.array([[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]], dtype=np.float64)
    return float(np.mean(np.linalg.norm(apply_homography(Ha, g) - apply_homography(Hb, g), axis=1)))


def frame_geometry(ref_pts, size=None):
    """Frame size (from the template's header when known, else approximated from the
    reference keypoints) and keypoint extent."""
    if size is not None:
        w, h = float(size[0]), float(size[1])
    else:
        w, h = float(ref_pts[:, 0].max()), float(ref_pts[:, 1].max())
    extent = np.maximum(np.ptp(ref_pts, axis=0), 100.0) if len(ref_pts) else np.array([w, h])
    return max(w, 1.0), max(h, 1.0), extent


# ------------------------------------------------------------------------------
# 8. Entry point
# ------------------------------------------------------------------------------
def part1(path1, path2, path3, path4, strategy="hybrid"):
    """part1(path_to_refdir, path_images_dir, path_feature_dir, path_output_dir, strategy='hybrid')"""
    refdir, images_dir, feat_dir, out_dir = path1, path2, path3, path4
    os.makedirs(out_dir, exist_ok=True)

    # [membro 2] Without the landmark file the old code raised here, before writing any
    # output; the brief requires H and Hc for every frame. H is still computed and Hc
    # is written as NaN: explicitly invalid rather than silently wrong. (membro 3 decides.)
    try:
        H_ref_to_court = load_court_anchor(refdir, images_dir)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"[part1] WARNING: no court anchor ({e})\n"
              "        H is still computed for every frame; Hc is written as NaN.")
        H_ref_to_court = None

    frames = list_frames(images_dir)
    if not frames:
        raise FileNotFoundError(f"No NAME_NNNN.jpg images in {images_dir}")
    print(f"[part1] {len(frames)} frames")
    feat_paths = [os.path.join(feat_dir, f"{name}.mat") for _, _, name in frames]

    # reference features (templateimg.mat); if absent, the first frame is the reference
    ref = (None, None)
    for d in (feat_dir, refdir):
        ref = try_load(os.path.join(d, "templateimg.mat"))
        if ref[0] is not None:
            break
    if ref[0] is None:                                    # first frame with features
        ref = next((f for f in map(try_load, feat_paths) if f[0] is not None), (None, None))
        print("[part1] no templateimg.mat: reference initialised from the first frame")

    # [membro 2] true frame size from the template header, and the overlay filter
    size = None
    for ext in IMG_EXTS:
        p = os.path.join(refdir, "templateimg" + ext)
        if os.path.isfile(p):
            try:
                size = image_size(p)
            except ValueError as e:
                print(f"[part1] could not read the size of {p}: {e}")
            break
    overlay = None
    if ref[0] is not None:
        overlay = find_static_overlay(ref, feat_paths, size)
        ref = drop_masked(ref, overlay)
        print(f"[part1] overlay filter: {overlay['n_keypoints']} static reference keypoints "
              f"removed, {len(ref[0])} left")
    geometry = frame_geometry(ref[0], size) if ref[0] is not None else None

    to_court = lambda H: H_ref_to_court @ H if H_ref_to_court is not None else np.full((3, 3), np.nan)
    prev, prev_idx = None, -1         # features (and index) of the last frame with a valid H
    H_prev = np.eye(3)                # its H
    last = None                       # features of the previous frame, valid H or not
    lost = False                      # after a cut, until a confirmed re-anchor
    pending = None                    # re-anchor candidate (idx, num_str, H, n) awaiting confirmation
    n_cons_hist, n_dir_hist = [], []  # inlier counts while tracking (cut / re-anchor thresholds)
    log = []

    for idx, (num, num_str, name) in enumerate(frames):
        cur = try_load(os.path.join(feat_dir, f"{name}.mat"), overlay)
        has_feat = cur[1] is not None

        if ref[0] is None and has_feat:                   # first frame becomes the reference
            ref = cur
            geometry = frame_geometry(ref[0])
            print(f"[part1] reference initialised from frame {num_str}")

        method, n_in, H, reason = "no_court", 0, None, "no_features"
        H_dir, n_dir, n_cons = None, 0, 0

        if has_feat and ref[0] is not None:
            w, h, extent = geometry

            # --- continuity with the previous frame: a cut breaks it
            H_cons = None
            if last is not None:
                H_cons, n_cons = estimate_pair(cur, last, extent, MIN_INLIERS_SEQ)
            if not lost and last is not None and is_cut(n_cons, n_cons_hist):
                lost, prev, pending = True, None, None
                print(f"[part1] cut at frame {num_str} ({n_cons} inliers to the previous frame, "
                      f"median {np.median(n_cons_hist[-CUT_WINDOW:]):.0f})")

            # --- direct: current -> reference
            H_dir, n_dir = estimate_pair(cur, ref, extent, MIN_INLIERS_DIRECT)
            if H_dir is not None and not is_plausible(H_dir, w, h):
                H_dir = None

            if lost:
                # --- re-anchor: a strong, rotation-consistent direct estimate, confirmed
                # by the next frame through the frame-to-frame H. Never by composition.
                ok = (H_dir is not None
                      and n_dir >= REANCHOR_FRAC * np.median(n_dir_hist or [0])
                      and rotation_residual(H_dir, w, h) <= ROT_MAX)
                if (ok and pending is not None and pending[0] == idx - 1 and H_cons is not None
                        and disagreement(H_dir, normalize_scale(pending[2] @ H_cons), w, h) <= CONFIRM_PX):
                    _, p_str, p_H, p_n = pending
                    save_homography(out_dir, p_str, p_H, to_court(p_H))
                    log[-1] = {"frame": log[-1]["frame"], "method": "reanchor", "inliers": p_n}
                    lost, pending = False, None
                    H, method, n_in = H_dir, "reanchor", n_dir
                    print(f"[part1] re-anchored at frame {p_str} ({p_n} inliers)")
                else:
                    pending = (idx, num_str, H_dir, n_dir) if ok else None
                    reason = "cut"
            else:
                # --- sequential: last good frame -> current, composed with its H
                H_seq = None
                if prev is not None:
                    H_rel, n_seq = ((H_cons, n_cons) if prev_idx == idx - 1
                                    else estimate_pair(cur, prev, extent, MIN_INLIERS_SEQ))
                    if H_rel is not None:
                        cand = normalize_scale(H_prev @ H_rel)
                        if is_plausible(cand, w, h):
                            H_seq = cand
                reason = "no_estimate"

            # --- choose based on strategy
            if lost or method == "reanchor":
                pass
            elif strategy == "direct":
                if H_dir is not None:
                    H, method, n_in = H_dir, "direct", n_dir
            elif strategy == "sequential":
                if prev is None:
                    H = H_dir if H_dir is not None else np.eye(3)
                    method, n_in = "sequential", n_dir if H_dir is not None else 0
                elif H_seq is not None:
                    H, method, n_in = H_seq, "sequential", n_seq
            else:  # "hybrid"
                if H_dir is not None and H_seq is not None:
                    if n_dir >= STRONG_INLIERS or disagreement(H_dir, H_seq, w, h) <= GATE_PX:
                        H, method, n_in = H_dir, "direct", n_dir
                    else:
                        H, method, n_in = H_seq, "sequential", n_seq
                elif H_dir is not None:
                    H, method, n_in = H_dir, "direct", n_dir
                elif H_seq is not None:
                    H, method, n_in = H_seq, "sequential", n_seq

        if H is None:                                     # cut / close-up / crowd / other camera
            H = np.full((3, 3), np.nan)                   # explicitly "no valid transform"
            entry = {"frame": num, "method": "no_court", "inliers": 0, "reason": reason}
        else:
            prev, prev_idx, H_prev = cur, idx, H.copy()
            if n_cons > 0:
                n_cons_hist.append(n_cons)
            if H_dir is not None:
                n_dir_hist.append(n_dir)
            entry = {"frame": num, "method": method, "inliers": n_in}
        last = cur if has_feat else None

        save_homography(out_dir, num_str, H, to_court(H))
        log.append(entry)

        if (idx + 1) % 50 == 0 or idx + 1 == len(frames):
            print(f"[part1] {idx + 1}/{len(frames)}  frame {num_str}: {method} ({n_in} inliers)")

    with open(os.path.join(out_dir, "tracking_log.json"), "w") as f:
        json.dump(log, f)

    counts = {m: sum(1 for e in log if e["method"] == m)
              for m in ("direct", "sequential", "reanchor", "no_court")}
    print(f"[part1] done: {counts}  -> {out_dir}")

    missing = missing_outputs(out_dir, frames)
    if missing:
        raise RuntimeError(f"no homography written for {len(missing)} frames: {missing[:5]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Part 1: Planar tracking & homography estimation")
    parser.add_argument("refdir", help="Path to reference directory (courtmodel.mat, templateimg.jpg)")
    parser.add_argument("images_dir", help="Path to image sequence folder")
    parser.add_argument("feat_dir", help="Path to SIFT features folder")
    parser.add_argument("out_dir", help="Output directory for homography_NNNN.mat")
    parser.add_argument("--strategy", choices=["hybrid", "direct", "sequential"], default="hybrid",
                        help="Tracking strategy: hybrid (default), direct, or sequential")
    args = parser.parse_args()
    part1(args.refdir, args.images_dir, args.feat_dir, args.out_dir, strategy=args.strategy)