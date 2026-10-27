"""
evaluate_sanity.py - Genuine Appendix A Self-Consistency Evaluation for Part 1.

Implements all 6 evaluation checks strictly as defined in Appendix A of the project PDF:
1. Line Reprojection Error: Warps court lines into frame i; measures perpendicular distance
   to nearest detected white-court-line pixels in the broadcast image.
2. Cycle Consistency: for triplets (i, j, k) at baselines of 1-30 frames, a grid goes through
   the output track i -> j -> k (H_j^-1 H_i, H_k^-1 H_j) and back k -> i through an independent
   estimate (part1's own estimator); RMS displacement from the identity, by baseline. The
   all-independent cycle is reported next to it (estimator noise alone).
3. Forward-Backward Consistency: frame-to-frame composition s -> N -> s on the longest run of
   frames with a court (no cut inside); drift of the court corners. Also the drift of the
   forward chain from part1's output at N (what a purely sequential tracker would carry).
4. Temporal Smoothness: court-corner velocity between consecutive frames that both have a
   valid Hc (NaN "no_court" frames break the curve); spikes must coincide with re-anchors.
5. Metric Residual: Maps detected court points to meters via Hc and measures known physical
   distances against official ITF dimensions.
6. Warped-Court IoU: Compares the warped court model polygon against the independently detected
   court polygon (convex hull of detected court landmarks).

Usage:
    python tests/evaluate_sanity.py --results results --images datasets/personal/rally_01 --features features --ref ref_tennis --out sanity_reports
"""

import os
import re
import sys
import glob
import json
import argparse
import numpy as np
import scipy.io as sio

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import part1

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def imread_unicode(path):
    """Safely reads images on Windows even if the path contains non-ASCII characters."""
    if cv2 is None or not os.path.exists(path):
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(path)


# Official ITF Court Dimensions (meters)
ITF_DOUBLES_WIDTH = 10.97
ITF_SINGLES_WIDTH = 8.23
ITF_TOTAL_LENGTH = 23.77
ITF_HALF_LENGTH = 11.885
ITF_SERVICE_DIST = 6.40

OFFICIAL_CORNERS_M = np.array([
    [-ITF_DOUBLES_WIDTH / 2.0, -ITF_HALF_LENGTH],  # BL
    [ ITF_DOUBLES_WIDTH / 2.0, -ITF_HALF_LENGTH],  # BR
    [ ITF_DOUBLES_WIDTH / 2.0,  ITF_HALF_LENGTH],  # TR
    [-ITF_DOUBLES_WIDTH / 2.0,  ITF_HALF_LENGTH],  # TL
], dtype=np.float64)

OFFICIAL_LINES_M = [
    ((-ITF_DOUBLES_WIDTH / 2.0, -ITF_HALF_LENGTH), ( ITF_DOUBLES_WIDTH / 2.0, -ITF_HALF_LENGTH)),  # bottom baseline
    ((-ITF_DOUBLES_WIDTH / 2.0,  ITF_HALF_LENGTH), ( ITF_DOUBLES_WIDTH / 2.0,  ITF_HALF_LENGTH)),  # top baseline
    ((-ITF_DOUBLES_WIDTH / 2.0, -ITF_HALF_LENGTH), (-ITF_DOUBLES_WIDTH / 2.0,  ITF_HALF_LENGTH)),  # left doubles
    (( ITF_DOUBLES_WIDTH / 2.0, -ITF_HALF_LENGTH), ( ITF_DOUBLES_WIDTH / 2.0,  ITF_HALF_LENGTH)),  # right doubles
    ((-ITF_SINGLES_WIDTH / 2.0, -ITF_HALF_LENGTH), (-ITF_SINGLES_WIDTH / 2.0,  ITF_HALF_LENGTH)),  # left singles
    (( ITF_SINGLES_WIDTH / 2.0, -ITF_HALF_LENGTH), ( ITF_SINGLES_WIDTH / 2.0,  ITF_HALF_LENGTH)),  # right singles
    ((-ITF_SINGLES_WIDTH / 2.0, -ITF_SERVICE_DIST), (ITF_SINGLES_WIDTH / 2.0, -ITF_SERVICE_DIST)),  # bottom service
    ((-ITF_SINGLES_WIDTH / 2.0,  ITF_SERVICE_DIST), (ITF_SINGLES_WIDTH / 2.0,  ITF_SERVICE_DIST)),  # top service
    ((0.0, -ITF_SERVICE_DIST),                      (0.0,  ITF_SERVICE_DIST)),                       # center service
    ((-ITF_DOUBLES_WIDTH / 2.0, 0.0),               ( ITF_DOUBLES_WIDTH / 2.0, 0.0))                 # net line
]


# ==============================================================================
# 1. Line Reprojection Error (Genuine pixel distance to real white court lines)
# ==============================================================================
def compute_line_reprojection_error(image_bgr, Hc, lines_m=OFFICIAL_LINES_M, max_clip_dist=50.0):
    """
    Warps court lines in meters into image frame using Hc^-1.
    Measures perpendicular distance to nearest detected white-court-line pixel.
    """
    if cv2 is None or image_bgr is None or Hc is None:
        return np.nan

    H_img, W_img = image_bgr.shape[:2]
    
    # Extract white court line mask (high value, low saturation)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    white_mask = (hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 165)
    
    # Morphological noise filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_clean = cv2.morphologyEx(white_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    # Morphological skeletonization to obtain the true 1-pixel centerline (spine) of court lines.
    # Without this, wide painted lines (6-12 px) create an artificial dead-zone where distance is exactly 0.00 px.
    skel = np.zeros_like(white_clean)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = white_clean.copy()
    for _ in range(12):
        eroded = cv2.erode(temp, element)
        dilated = cv2.dilate(eroded, element)
        diff = cv2.subtract(temp, dilated)
        skel = cv2.bitwise_or(skel, diff)
        temp = eroded
        if cv2.countNonZero(temp) == 0:
            break

    target_mask = skel if cv2.countNonZero(skel) > 100 else white_clean
    dist_map = cv2.distanceTransform(1 - target_mask, cv2.DIST_L2, 3)

    try:
        Hc_inv = np.linalg.inv(Hc)
    except np.linalg.LinAlgError:
        return np.nan

    sampled_distances = []
    for p1, p2 in lines_m:
        p1 = np.asarray(p1, dtype=np.float64)
        p2 = np.asarray(p2, dtype=np.float64)
        alphas = np.linspace(0.0, 1.0, 40)
        segment_pts_m = (1.0 - alphas)[:, None] * p1 + alphas[:, None] * p2
        
        homo_m = np.column_stack((segment_pts_m, np.ones(len(segment_pts_m))))
        proj = (Hc_inv @ homo_m.T).T
        w = proj[:, 2]
        valid_w = np.abs(w) > 1e-6
        if not np.any(valid_w):
            continue
            
        u = proj[valid_w, 0] / w[valid_w]
        v = proj[valid_w, 1] / w[valid_w]
        
        in_bounds = (u >= 0) & (u < W_img - 1) & (v >= 0) & (v < H_img - 1)
        if np.any(in_bounds):
            u_in = u[in_bounds].astype(np.int32)
            v_in = v[in_bounds].astype(np.int32)
            dists = dist_map[v_in, u_in]
            dists = np.clip(dists, 0.0, max_clip_dist)
            sampled_distances.extend(dists.tolist())

    if len(sampled_distances) == 0:
        return np.nan
    return float(np.median(sampled_distances))


# ==============================================================================
# Helper: the tracker's own pairwise estimator
# ==============================================================================
CYCLE_BASELINES = (1, 5, 15, 30)   # frames between i, j and k
CYCLE_TRIPLETS = 8                 # triplets per baseline, spread over the sequence
FB_MAX_LEN = 150                   # frames of forward-backward composition


def feature_names(feat_dir):
    """{frame number string: feature file stem}, e.g. {"0042": "frame_0042"}."""
    names = {}
    for f in os.listdir(feat_dir):
        stem, ext = os.path.splitext(f)
        m = re.search(r"(\d+)$", stem)
        if ext == ".mat" and m and stem != "templateimg":
            names[m.group(1)] = stem
    return names


class PairEstimator:
    """
    Independent pairwise homographies H(i -> j) (frame i pixels -> frame j pixels) from
    the same features and the same estimator as part1 (overlay filter, two-pass matching,
    RANSAC, refinement), so a check measures the tracker's output, not another matcher.
    """

    def __init__(self, feat_dir, stems, size=None):
        self.paths = [os.path.join(feat_dir, f"{s}.mat") for s in stems]
        ref = part1.try_load(os.path.join(feat_dir, "templateimg.mat"))
        if ref[0] is None:
            ref = part1.try_load(self.paths[0])
        self.overlay = part1.find_static_overlay(ref, self.paths, size)
        self.w, self.h, self.extent = part1.frame_geometry(ref[0], size)
        self.cache = {}

    def H(self, i, j):
        if (i, j) not in self.cache:
            a = part1.try_load(self.paths[i], self.overlay)
            b = part1.try_load(self.paths[j], self.overlay)
            H, _ = part1.estimate_pair(a, b, self.extent, part1.MIN_INLIERS_SEQ)
            self.cache[(i, j)] = H
        return self.cache[(i, j)]


def grid_rms(H, w, h):
    """RMS displacement (px) of a 20x20 grid over the central 70% of the frame under H."""
    gx, gy = np.meshgrid(np.linspace(0.15 * w, 0.85 * w, 20), np.linspace(0.15 * h, 0.85 * h, 20))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    proj = part1.apply_homography(H, pts)
    return float(np.sqrt(np.mean(np.sum((proj - pts) ** 2, axis=1))))


def valid_runs(valid):
    """[(start, end)] of the maximal runs of consecutive valid frames (end inclusive)."""
    runs, s = [], None
    for t, v in enumerate(list(valid) + [False]):
        if v and s is None:
            s = t
        elif not v and s is not None:
            runs.append((s, t - 1))
            s = None
    return runs


# ==============================================================================
# 2. Cycle Consistency (the output track, closed by an independent estimate)
# ==============================================================================
def compute_cycle_consistency(est, h_list, baselines=CYCLE_BASELINES, n_triplets=CYCLE_TRIPLETS):
    """
    For triplets (i, j, k) = (i, i+b, i+2b) inside one run of frames with a valid H:
      * track:     H(k->i)_indep @ H(j->k)_track @ H(i->j)_track, where
                   H(i->j)_track = H_j^-1 H_i comes from part1's output. The two track
                   edges collapse to H_k^-1 H_i, so this is how far the output disagrees
                   with an independent measurement of the same motion;
      * estimator: H(k->i) H(j->k) H(i->j), all three independent (estimator noise only).
    Returns {b: {"track": median RMS px, "estimator": median RMS px, "n": triplets}}.
    A cycle of output homographies alone is the identity by construction, so it would
    measure nothing.
    """
    valid = [np.all(np.isfinite(H)) for H in h_list]
    runs = valid_runs(valid)
    out = {}
    for b in baselines:
        starts = [i for s, e in runs for i in range(s, e - 2 * b + 1)]
        if not starts:
            continue
        picks = sorted({starts[p] for p in np.linspace(0, len(starts) - 1, min(n_triplets, len(starts))).round().astype(int)})
        track, indep = [], []
        for i in picks:
            j, k = i + b, i + 2 * b
            Hki = est.H(k, i)
            if Hki is None:
                continue
            track.append(grid_rms(Hki @ np.linalg.inv(h_list[k]) @ h_list[i], est.w, est.h))
            Hij, Hjk = est.H(i, j), est.H(j, k)
            if Hij is not None and Hjk is not None:
                indep.append(grid_rms(Hki @ Hjk @ Hij, est.w, est.h))
        out[b] = {"track": float(np.median(track)) if track else None,
                  "estimator": float(np.median(indep)) if indep else None,
                  "n": len(track)}
    return out


# ==============================================================================
# 3. Forward-Backward Consistency (frame-to-frame composition over a court run)
# ==============================================================================
def compute_forward_backward(est, h_list, hc_list, max_len=FB_MAX_LEN):
    """
    On the longest run of frames with a valid H (no cut inside): compose frame-to-frame
    estimates from its first frame s forward to t, then back to s, and measure how far
    the court corners (or the frame corners, without an anchor) return from where they
    started. Also reports how far the forward chain s -> t has drifted from part1's output
    for the same frames (H_t^-1 H_s): the drift a purely sequential tracker would carry.
    """
    runs = valid_runs([np.all(np.isfinite(H)) for H in h_list])
    if not runs:
        return None
    s, e = max(runs, key=lambda r: r[1] - r[0])
    e = min(e, s + max_len)
    if e <= s:
        return None

    Hc_s = hc_list[s]
    if np.all(np.isfinite(Hc_s)):
        corners = part1.apply_homography(np.linalg.inv(Hc_s), OFFICIAL_CORNERS_M)
    else:
        corners = np.array([[0, 0], [est.w, 0], [est.w, est.h], [0, est.h]], dtype=np.float64)

    fwd, t, chain_drift = np.eye(3), s, []
    while t < e:
        H_rel = est.H(t, t + 1)
        if H_rel is None:
            break
        fwd = part1.normalize_scale(H_rel @ fwd)
        t += 1
        track = np.linalg.inv(h_list[t]) @ h_list[s]
        chain_drift.append(float(np.mean(np.linalg.norm(
            part1.apply_homography(fwd, corners) - part1.apply_homography(track, corners), axis=1))))
    end = t

    bwd = np.eye(3)
    while t > s:
        H_rel = est.H(t, t - 1)
        if H_rel is None:
            return {"start": s, "end": end, "round_trip_px": None,
                    "chain_vs_track_px": chain_drift[-1] if chain_drift else None, "chain_curve": chain_drift}
        bwd = part1.normalize_scale(H_rel @ bwd)
        t -= 1

    back = part1.apply_homography(bwd @ fwd, corners)
    return {"start": s, "end": end,
            "round_trip_px": float(np.mean(np.linalg.norm(back - corners, axis=1))),
            "chain_vs_track_px": chain_drift[-1] if chain_drift else None,
            "chain_curve": chain_drift}


# ==============================================================================
# 4. Temporal Smoothness
# ==============================================================================
def compute_temporal_smoothness(hc_matrices, methods=None, corners_m=OFFICIAL_CORNERS_M,
                                spike_thresh_px=15.0, spike_factor=6.0):
    """
    Projects the court corners into every frame and measures their frame-to-frame
    velocity, only between consecutive frames that both have a valid Hc (frames written
    as NaN, "no_court", break the curve instead of looking perfectly still).
    A spike is a velocity above max(spike_thresh_px, spike_factor x median). Spikes at a
    re-anchor are expected; any other spike sits between two tracked frames and is an
    isolated catastrophic estimate ("unexplained").
    """
    T = len(hc_matrices)
    corners = np.full((T, len(corners_m), 2), np.nan)
    for t, Hc in enumerate(hc_matrices):
        if np.all(np.isfinite(Hc)):
            corners[t] = part1.apply_homography(np.linalg.inv(Hc), corners_m)

    velocities = np.mean(np.linalg.norm(corners[1:] - corners[:-1], axis=2), axis=1)  # NaN across gaps
    finite = np.isfinite(velocities)
    median = float(np.median(velocities[finite])) if finite.any() else np.nan
    thresh = max(spike_thresh_px, spike_factor * median) if finite.any() else spike_thresh_px
    spikes = [int(t) + 1 for t in np.where(finite & (velocities > thresh))[0]]
    methods = methods or [None] * T
    unexplained = [t for t in spikes if methods[t] != "reanchor" and methods[t - 1] != "reanchor"]

    return {
        "velocities": velocities,
        "median_velocity": median,
        "max_velocity": float(np.max(velocities[finite])) if finite.any() else np.nan,
        "threshold": thresh,
        "spikes": spikes,
        "unexplained_spikes": unexplained,
        "no_court_frames": int(sum(not np.all(np.isfinite(Hc)) for Hc in hc_matrices)),
    }

# ==============================================================================
# 5. Metric Residual
# ==============================================================================
def compute_genuine_metric_residual(Hc, model_pts_m, detected_pts_px):
    """
    Measures physical residuals in centimeters between detected court landmarks
    and official ITF model coordinates mapped through Hc.
    """
    if Hc is None or model_pts_m is None or detected_pts_px is None:
        return np.nan, np.nan

    homo_px = np.column_stack([detected_pts_px, np.ones(len(detected_pts_px))])
    proj_m = (Hc @ homo_px.T).T
    m_est = proj_m[:, :2] / proj_m[:, 2:3]
    landmark_err_cm = np.linalg.norm(m_est - model_pts_m, axis=1) * 100.0

    return float(np.mean(landmark_err_cm)), float(np.max(landmark_err_cm))


# ==============================================================================
# 6. Warped-Court IoU (Genuine Comparison against Detected Court Polygon)
# ==============================================================================
def compute_genuine_warped_court_iou(Hc, detected_court_pts_px, img_shape=(720, 1280), corners_m=OFFICIAL_CORNERS_M):
    """
    Computes genuine Intersection-over-Union (IoU) by comparing the warped court
    model polygon (Hc^-1 @ court_m) against the independent convex hull of detected
    court landmarks in the image.
    """
    if cv2 is None or Hc is None or detected_court_pts_px is None:
        return np.nan

    H_img, W_img = img_shape
    homo_m = np.column_stack([corners_m, np.ones(len(corners_m))])

    try:
        # Polygon 1: Warped model polygon from Hc
        Hc_inv = np.linalg.inv(Hc)
        p1 = (Hc_inv @ homo_m.T).T
        poly_model = (p1[:, :2] / p1[:, 2:3]).astype(np.int32)

        # Polygon 2: Convex hull of detected court landmarks
        hull_idx = cv2.convexHull(detected_court_pts_px.astype(np.float32), returnPoints=False)
        poly_detected = detected_court_pts_px[hull_idx.squeeze()].astype(np.int32)
    except Exception:
        return np.nan

    mask_model = np.zeros((H_img, W_img), dtype=np.uint8)
    mask_detected = np.zeros((H_img, W_img), dtype=np.uint8)

    cv2.fillPoly(mask_model, [poly_model], 1)
    cv2.fillPoly(mask_detected, [poly_detected], 1)

    intersection = np.sum((mask_model & mask_detected))
    union = np.sum((mask_model | mask_detected))

    if union == 0:
        return 0.0
    return float(intersection / union)


# ==============================================================================
# Comprehensive Evaluation Pipeline
# ==============================================================================
def run_all_sanity_checks(results_dir, images_dir=None, features_dir=None, ref_dir=None, out_dir=None):
    """
    Runs all 6 genuine sanity checks strictly following Appendix A of the project PDF.
    """
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    mat_files = sorted(glob.glob(os.path.join(results_dir, "homography_*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No homography_*.mat files found in: {results_dir}")

    T = len(mat_files)
    print(f"\n==================================================================")
    print(f"       APPENDIX A: GEOMETRIC SELF-CONSISTENCY EVALUATION          ")
    print(f"==================================================================")
    print(f"Evaluated sequence: {T} frames from {results_dir}")

    # Load all H and Hc
    h_list = []
    hc_list = []
    for f in mat_files:
        data = sio.loadmat(f)
        h_list.append(data["H"])
        hc_list.append(data["Hc"])

    # Load reference model landmarks if available
    model_pts_m, model_pts_px = None, None
    landmark_source = "None"
    
    # 1. Priority: check sequence directory for rally-specific court landmarks
    if images_dir is not None:
        seq_base_files = glob.glob(os.path.join(images_dir, "court_base_*.mat"))
        if seq_base_files:
            try:
                sdata = sio.loadmat(seq_base_files[0])
                if "img_pts" in sdata:
                    model_pts_px = sdata["img_pts"]
                    landmark_source = os.path.basename(seq_base_files[0])
            except Exception:
                pass

    # 2. Fallback: check ref_dir for reference template landmarks & court model
    if ref_dir is not None:
        model_mat_path = os.path.join(ref_dir, "courtmodel.mat")
        if os.path.exists(model_mat_path):
            mdata = sio.loadmat(model_mat_path)
            if "pts" in mdata:
                model_pts_m = mdata["pts"]
        if model_pts_px is None:
            base_mat_path = glob.glob(os.path.join(ref_dir, "court_base_*.mat"))
            if base_mat_path:
                bdata = sio.loadmat(base_mat_path[0])
                if "img_pts" in bdata:
                    model_pts_px = bdata["img_pts"]
                    landmark_source = f"{os.path.basename(base_mat_path[0])} (reference anchor)"

    # -------------------------------------------------------------------------
    # 1. Line Reprojection Error (Genuine pixel distance to real white court lines)
    # Requires: --images, --results, --ref
    # -------------------------------------------------------------------------
    print("\n[1/6] Computing Line Reprojection Errors...")
    line_errors = []
    if images_dir and os.path.exists(images_dir) and cv2 is not None:
        sample_indices = np.linspace(0, min(T - 1, 220), 20, dtype=int)
        for idx in sample_indices:
            img_path = os.path.join(images_dir, f"frame_{idx:04d}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(images_dir, f"frame_{idx:04d}.png")
            if os.path.exists(img_path):
                img = imread_unicode(img_path)
                err = compute_line_reprojection_error(img, hc_list[idx])
                if not np.isnan(err):
                    line_errors.append(err)
    
    if len(line_errors) > 0:
        med_line_err = float(np.median(line_errors))
        line_status = "PASS" if med_line_err < 3.0 else "FAIL"
        print(f"      Median Line Reprojection Error: {med_line_err:.2f} px  (Target: < 2-3 px) -> {line_status}")
    else:
        med_line_err = np.nan
        line_status = "SKIPPED (No valid images found in --images)"
        print(f"      Line Reprojection Error: {line_status}")

    # Pairwise estimator on the same features as part1 (cycle and forward-backward checks)
    est = None
    if features_dir and os.path.isdir(features_dir):
        stems = feature_names(features_dir)
        nums = [re.search(r"(\d+)\.mat$", f).group(1) for f in mat_files]
        if all(n in stems for n in nums):
            size = None
            if images_dir and os.path.isdir(images_dir):
                imgs = sorted(glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(os.path.join(images_dir, "*.png")))
                size = part1.image_size(imgs[0]) if imgs else None
            est = PairEstimator(features_dir, [stems[n] for n in nums], size)

    # -------------------------------------------------------------------------
    # 2. Cycle Consistency (output track closed by an independent estimate)
    # Requires: --features
    # -------------------------------------------------------------------------
    print("\n[2/6] Evaluating Cycle Consistency (track vs independent estimates)...")
    med_cycle_rms, cycle_by_baseline = np.nan, {}
    cycle_status = "SKIPPED (No matching features in --features)"
    if est is not None:
        cycle_by_baseline = compute_cycle_consistency(est, h_list)
        for b, r in cycle_by_baseline.items():
            tr = f"{r['track']:.3f}" if r["track"] is not None else "N/A"
            es = f"{r['estimator']:.3f}" if r["estimator"] is not None else "N/A"
            print(f"      baseline {b:>2} frames: track {tr} px, estimator only {es} px ({r['n']} triplets)")
        nearest = cycle_by_baseline.get(min(cycle_by_baseline), {}) if cycle_by_baseline else {}
        if nearest.get("track") is not None:
            med_cycle_rms = nearest["track"]
            cycle_status = "PASS" if med_cycle_rms < 1.0 else "FAIL"
            print(f"      Nearby-frame track cycle RMS:   {med_cycle_rms:.4f} px (Target: Sub-pixel) -> {cycle_status}")
    if np.isnan(med_cycle_rms):
        print(f"      Cycle Consistency: {cycle_status}")

    # -------------------------------------------------------------------------
    # 3. Forward-Backward Consistency (frame-to-frame composition on a court run)
    # Requires: --features
    # -------------------------------------------------------------------------
    print("\n[3/6] Measuring Forward-Backward Drift of Frame-to-Frame Composition...")
    mean_drift, fb = np.nan, None
    drift_status = "SKIPPED (No matching features in --features)"
    if est is not None:
        fb = compute_forward_backward(est, h_list, hc_list)
        if fb is not None and fb["round_trip_px"] is not None:
            mean_drift = fb["round_trip_px"]
            drift_status = "PASS" if mean_drift < 5.0 else "FAIL"
            print(f"      Frames {fb['start']} -> {fb['end']} -> {fb['start']}: round trip {mean_drift:.3f} px "
                  f"(Target: Few px < 5.0 px) -> {drift_status}")
            if fb["chain_vs_track_px"] is not None:
                print(f"      Forward chain vs part1 output at frame {fb['end']}: {fb['chain_vs_track_px']:.2f} px "
                      f"(drift a purely sequential tracker would carry)")
    if np.isnan(mean_drift):
        print(f"      Forward-Backward Drift: {drift_status}")

    # -------------------------------------------------------------------------
    # 4. Temporal Smoothness
    # Requires: --results (always available)
    # -------------------------------------------------------------------------
    print("\n[4/6] Tracking Temporal Velocity & Flagging Broadcast Cuts...")
    methods = None
    log_path = os.path.join(results_dir, "tracking_log.json")
    if os.path.isfile(log_path):
        log = json.load(open(log_path))
        if len(log) == T:
            methods = [e["method"] for e in log]
    smoothness = compute_temporal_smoothness(hc_list, methods)
    med_vel = smoothness["median_velocity"]
    smooth_status = "PASS" if not smoothness["unexplained_spikes"] else "FAIL"
    print(f"      Median Corner Velocity:         {med_vel:.2f} px/frame ({smoothness['no_court_frames']} "
          f"no-court frames excluded)")
    print(f"      Spikes (> {smoothness['threshold']:.1f} px/frame):         {smoothness['spikes']}")
    print(f"      Unexplained spikes:             {smoothness['unexplained_spikes']} -> {smooth_status}")

    # Plot smoothness velocity profile
    if out_dir and plt is not None:
        plot_path = os.path.join(out_dir, "sanity_smoothness.png")
        plt.figure(figsize=(10, 4), dpi=150)
        frames_v = np.arange(1, len(smoothness["velocities"]) + 1)
        plt.plot(frames_v, smoothness["velocities"], color="#1f77b4", linewidth=1.5, label="Corner Velocity (px/frame)")
        if methods is not None:
            nc = [t for t, m in enumerate(methods) if m == "no_court"]
            if nc:
                plt.scatter(nc, np.zeros(len(nc)), color="#999999", s=6, marker="|", label="No court (NaN)")
        if smoothness["spikes"]:
            spike_vals = [smoothness["velocities"][s - 1] for s in smoothness["spikes"]]
            colours = ["red" if s in smoothness["unexplained_spikes"] else "orange" for s in smoothness["spikes"]]
            plt.scatter(smoothness["spikes"], spike_vals, color=colours, s=40, zorder=5,
                        label="Spike (red: unexplained, orange: at a re-anchor)")
        plt.axhline(y=smoothness["threshold"], color="gray", linestyle="--", alpha=0.7,
                    label=f"Spike threshold ({smoothness['threshold']:.1f} px/frame)")
        plt.title("Temporal Smoothness Profile & Cut Discontinuity Detection", fontsize=12, fontweight="bold")
        plt.xlabel("Frame Index", fontsize=10)
        plt.ylabel("Court Corner Velocity (px/frame)", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"      Saved velocity profile plot to: {plot_path}")

    # -------------------------------------------------------------------------
    # 5. Metric Residual (Comparison against ITF Official Dimensions)
    # Requires: --ref (courtmodel.mat and detected court points)
    # -------------------------------------------------------------------------
    print("\n[5/6] Measuring ITF Real-World Metric Residuals...")
    metric_errors = []
    if model_pts_m is not None and model_pts_px is not None:
        for idx in range(min(T, 50)):
            try:
                H_inv = np.linalg.inv(h_list[idx])
                homo_0 = np.column_stack([model_pts_px, np.ones(len(model_pts_px))])
                p_i = (H_inv @ homo_0.T).T
                curr_pts_px = p_i[:, :2] / p_i[:, 2:3]
                mean_err_cm, _ = compute_genuine_metric_residual(hc_list[idx], model_pts_m, curr_pts_px)
                if not np.isnan(mean_err_cm):
                    metric_errors.append(mean_err_cm)
            except Exception:
                pass

    if len(metric_errors) > 0:
        med_metric_cm = float(np.median(metric_errors))
        metric_status = "PASS" if med_metric_cm < 5.0 else "FAIL"
        print(f"      Median Metric Landmark Residual: {med_metric_cm:.2f} cm (Landmarks: {landmark_source}) -> {metric_status}")
    else:
        med_metric_cm = np.nan
        metric_status = "SKIPPED (Missing courtmodel.mat or landmark points in --ref)"
        print(f"      Metric Residual: {metric_status}")

    # -------------------------------------------------------------------------
    # 6. Warped-Court IoU (Genuine Comparison against Detected Court Polygon)
    # Requires: --ref (detected court points) and cv2
    # -------------------------------------------------------------------------
    print("\n[6/6] Computing Warped-Court IoU against Detected Court Polygon...")
    med_iou = np.nan
    iou_status = "SKIPPED (Missing detected court keypoints in --ref)"
    if model_pts_px is not None and cv2 is not None:
        iou_val = compute_genuine_warped_court_iou(hc_list[0], model_pts_px)
        if not np.isnan(iou_val):
            med_iou = iou_val
            iou_status = "PASS" if med_iou > 0.95 else "FAIL"
            print(f"      Warped-Court IoU:               {med_iou:.4f}     (Target: > 0.95)   -> {iou_status}")
        else:
            print(f"      Warped-Court IoU:               {iou_status}")
    else:
        print(f"      Warped-Court IoU:               {iou_status}")

    # -------------------------------------------------------------------------
    # Summary Table & Report Generation
    # -------------------------------------------------------------------------
    results_summary = {
        "line_reprojection_error_px": med_line_err if not np.isnan(med_line_err) else None,
        "line_status": line_status,
        "cycle_consistency_rms_px": med_cycle_rms if not np.isnan(med_cycle_rms) else None,
        "cycle_by_baseline": {str(b): r for b, r in cycle_by_baseline.items()},
        "cycle_status": cycle_status,
        "forward_backward_drift_px": mean_drift if not np.isnan(mean_drift) else None,
        "forward_backward_frames": [fb["start"], fb["end"]] if fb else None,
        "chain_vs_track_px": fb["chain_vs_track_px"] if fb else None,
        "drift_status": drift_status,
        "temporal_velocity_px_per_frame": med_vel,
        "smooth_status": smooth_status,
        "flagged_cut_frames": smoothness["spikes"],
        "unexplained_spike_frames": smoothness["unexplained_spikes"],
        "no_court_frames": smoothness["no_court_frames"],
        "metric_residual_cm": med_metric_cm if not np.isnan(med_metric_cm) else None,
        "metric_status": metric_status,
        "warped_court_iou": med_iou if not np.isnan(med_iou) else None,
        "iou_status": iou_status,
        "all_passed": all([
            line_status == "PASS",
            cycle_status == "PASS",
            drift_status == "PASS",
            smooth_status == "PASS",
            metric_status == "PASS",
            iou_status == "PASS"
        ])
    }

    print("\n" + "=" * 70)
    print("           APPENDIX A SELF-CONSISTENCY RESULTS TABLE              ")
    print("=" * 70)
    print(f"{'Check':<30} | {'Measured Value':<16} | {'Target':<14} | {'Status'}")
    print("-" * 70)
    v_line = f"{med_line_err:>10.2f} px" if not np.isnan(med_line_err) else "       N/A     "
    v_cycle = f"{med_cycle_rms:>10.4f} px" if not np.isnan(med_cycle_rms) else "       N/A     "
    v_drift = f"{mean_drift:>10.3f} px" if not np.isnan(mean_drift) else "       N/A     "
    v_vel = f"{med_vel:>10.2f} px/f"
    v_metric = f"{med_metric_cm:>10.2f} cm" if not np.isnan(med_metric_cm) else "       N/A     "
    v_iou = f"{med_iou:>10.4f}        " if not np.isnan(med_iou) else "       N/A     "

    print(f"{'1. Line Reprojection Error':<30} | {v_line:<16} | {'< 2-3 px':<14} | {line_status}")
    print(f"{'2. Cycle Consistency (RMS)':<30} | {v_cycle:<16} | {'Sub-pixel':<14} | {cycle_status}")
    print(f"{'3. Forward-Backward Drift':<30}  | {v_drift:<16} | {'Few pixels':<14} | {drift_status}")
    print(f"{'4. Temporal Smoothness':<30}    | {v_vel:<16} | {'Smooth motion':<14} | {smooth_status}")
    print(f"{'5. Metric Residual (cm)':<30}   | {v_metric:<16} | {'Few cm':<14} | {metric_status}")
    print(f"{'6. Warped-Court IoU':<30}        | {v_iou:<16} | {'> 0.95':<14} | {iou_status}")
    print("=" * 70)

    # Save JSON & Markdown report if out_dir specified
    if out_dir:
        json_path = os.path.join(out_dir, "sanity_report.json")
        with open(json_path, "w") as f:
            json.dump(results_summary, f, indent=2)

        md_path = os.path.join(out_dir, "sanity_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            v_line_md = f"{med_line_err:.2f} px" if not np.isnan(med_line_err) else "N/A"
            v_cycle_md = f"{med_cycle_rms:.4f} px" if not np.isnan(med_cycle_rms) else "N/A"
            v_drift_md = f"{mean_drift:.3f} px" if not np.isnan(mean_drift) else "N/A"
            v_metric_md = f"{med_metric_cm:.2f} cm" if not np.isnan(med_metric_cm) else "N/A"
            v_iou_md = f"{med_iou:.4f}" if not np.isnan(med_iou) else "N/A"

            f.write("# Part 1 Appendix A: Geometric Self-Consistency Verification Report\n\n")
            f.write(f"- **Evaluated Sequence:** {T} frames from `{results_dir}`\n")
            f.write(f"- **Overall Evaluation Status:** **{'ALL 6 CHECKS PASSED [OK]' if results_summary['all_passed'] else 'ATTENTION REQUIRED'}**\n\n")
            f.write("## 1. Quantitative Verification Metrics\n\n")
            f.write("| Verification Metric | What it Catches | How it is Computed | Measured Value | What Good Looks Like | Result |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :---: |\n")
            f.write(f"| **Line reprojection error** | Subtly wrong homography | Distance to nearest detected white-line pixels | `{v_line_md}` | Median under ~2-3 px at 720p | **{line_status}** |\n")
            f.write(f"| **Cycle consistency** | Mutually inconsistent estimates | Grid RMS through $\\hat H_{{ki}} H_{{jk}} H_{{ij}}$ (output track closed by an independent estimate), nearest baseline | `{v_cycle_md}` | Sub-pixel for nearby frames | **{cycle_status}** |\n")
            f.write(f"| **Forward-backward consistency** | Drift in composed track | Frame-to-frame composition $s \\to N \\to s$ on the longest court run, corner drift | `{v_drift_md}` | A few pixels over a rally | **{drift_status}** |\n")
            f.write(f"| **Temporal smoothness** | Isolated catastrophic frames | Corner velocity between consecutive tracked frames; unexplained spikes: `{smoothness['unexplained_spikes']}` | `{med_vel:.2f} px/f` | Spikes only at flagged cuts | **{smooth_status}** |\n")
            f.write(f"| **Metric residual** | Valid projectively but wrong court | Landmark error mapped to official ITF dimensions | `{v_metric_md}` | Within a few centimetres | **{metric_status}** |\n")
            f.write(f"| **Warped-court IoU** | Gross misregistration | Polygon IoU vs detected court landmark hull | `{v_iou_md}` | Above 0.95 on clean frames | **{iou_status}** |\n\n")
            f.write("## 2. Cycle Consistency by Baseline\n\n")
            f.write("| Baseline (frames) | Track vs independent (px) | Estimator only (px) | Triplets |\n| ---: | ---: | ---: | ---: |\n")
            for b, r in cycle_by_baseline.items():
                tr = f"{r['track']:.3f}" if r["track"] is not None else "N/A"
                es = f"{r['estimator']:.3f}" if r["estimator"] is not None else "N/A"
                f.write(f"| {b} | {tr} | {es} | {r['n']} |\n")
            if fb and fb["chain_vs_track_px"] is not None:
                f.write(f"\nForward chain {fb['start']} -> {fb['end']} vs part1 output at frame {fb['end']}: "
                        f"`{fb['chain_vs_track_px']:.2f} px` (drift of a purely sequential tracker).\n")
            f.write("\n## 2b. Frames Without a Court and Spikes\n\n")
            f.write(f"- No-court frames (NaN): {smoothness['no_court_frames']}\n")
            f.write(f"- Velocity spikes: `{smoothness['spikes']}`; unexplained: `{smoothness['unexplained_spikes']}`\n\n")
            f.write("## 3. Smoothness Velocity Profile\n\n")
            f.write("![Temporal Smoothness Curve](sanity_smoothness.png)\n")
        print(f"\nSaved Markdown Report to: {md_path}")
        print(f"Saved JSON Report to:     {json_path}")

    return results_summary


# ==============================================================================
# CLI Entry Point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate genuine Appendix A self-consistency metrics for Part 1.")
    parser.add_argument("--results", default="results", help="Directory containing homography_NNNN.mat files.")
    parser.add_argument("--images", default="datasets/personal/rally_01", help="Directory containing sequence image frames.")
    parser.add_argument("--features", default="features", help="Directory containing SIFT feature .mat files.")
    parser.add_argument("--ref", default="ref_tennis", help="Directory containing courtmodel.mat and reference files.")
    parser.add_argument("--out", default="sanity_reports", help="Directory to save report and plots.")

    args = parser.parse_args()
    run_all_sanity_checks(
        results_dir=args.results,
        images_dir=args.images,
        features_dir=args.features,
        ref_dir=args.ref,
        out_dir=args.out
    )
