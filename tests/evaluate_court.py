#!/usr/bin/env python3
"""
evaluate_court.py - court-based self-consistency metrics for Part I (Appendix A).

Computes, per frame, the three metrics that depend on the metric court anchor Hc:

  1. Line reprojection error  - warp the ITF court lines into frame i through
     inv(Hc) and measure the distance from each warped sample to the nearest
     detected white-line pixel (distance transform). Good: median < 2-3 px at
     720p on clean frames.
  5. Metric residual          - re-fit the painted lines from the white pixels,
     intersect them to measure the court landmarks in pixels, map those through
     Hc to metres, and compare known ITF distances. Good: within a few cm.
  6. Warped-court IoU         - for hand-annotated frames (tests/annotate_court.py),
     overlap of the hand-marked doubles polygon with the model polygon warped by
     inv(Hc). Good: > 0.95 on clean frames.

(The other three Appendix A metrics - cycle consistency, forward-backward
consistency, temporal smoothness - depend only on H, not the court anchor; this
script is laid out so they can be dropped in alongside later.)

Usage
-----
    uv run python tests/evaluate_court.py \
        --results results/rally_04 \
        --images  datasets/rally_04 \
        --ref     ref/rally_04 \
        --out     court_reports/rally_04
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob                                                        # noqa: E402
from main1 import list_images, frame_number                       # noqa: E402
from court_lines import (load_court_model, project_lines, project_points,   # noqa: E402
                         white_line_mask, polyline_proximity_mask,
                         fit_line_tls, intersect_lines, ITF_DISTANCES)

LINE_ERR_TRUNC = 20.0     # px: cap the line reprojection error (ignore far-away junk)
METRIC_BAND = 8.0         # px: half-width of the band used to re-fit each line
CLEAN_MIN_INFRAME = 0.40  # a frame is "clean" if >= this fraction of samples is in-frame
IOU_PAD = 300             # px padding so court corners outside the frame still rasterise


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_hc(results_dir, num_str):
    path = os.path.join(results_dir, f"homography_{num_str}.mat")
    if not os.path.isfile(path):
        return None
    Hc = np.asarray(sio.loadmat(path)["Hc"], dtype=np.float64).reshape(3, 3)
    return Hc if np.all(np.isfinite(Hc)) else None


def load_methods(results_dir):
    methods, inliers = {}, {}
    log = os.path.join(results_dir, "tracking_log.json")
    if os.path.isfile(log):
        with open(log) as f:
            for e in json.load(f):
                methods[e["frame"]] = e.get("method", "?")
                inliers[e["frame"]] = e.get("inliers", 0)
    return methods, inliers


# ---------------------------------------------------------------------------
# metric 1: line reprojection error
# ---------------------------------------------------------------------------
def line_reprojection_error(dist_map, polylines):
    """
    Distance (px, truncated) from each in-frame warped line sample to the nearest
    detected white pixel. Returns (frame_median, frame_mean, per_line_median,
    in_frame_fraction). The median saturates at 0 when the warped line sits
    within the painted line's width (what good looks like); the mean still shows
    sub-pixel structure, so both are reported.
    """
    h, w = dist_map.shape
    per_line, pooled = [], []
    n_total = n_in = 0
    for pl in polylines:
        xy, valid = pl["pts"], pl["valid"]
        n_total += len(valid)
        n_in += int(valid.sum())
        if not valid.any():
            per_line.append(np.nan)
            continue
        px = np.round(xy[valid]).astype(int)
        px[:, 0] = np.clip(px[:, 0], 0, w - 1)
        px[:, 1] = np.clip(px[:, 1], 0, h - 1)
        d = np.minimum(dist_map[px[:, 1], px[:, 0]], LINE_ERR_TRUNC)
        per_line.append(float(np.median(d)))
        pooled.extend(d.tolist())
    frame_med = float(np.median(pooled)) if pooled else np.nan
    frame_mean = float(np.mean(pooled)) if pooled else np.nan
    frac = n_in / n_total if n_total else 0.0
    return frame_med, frame_mean, per_line, frac


# ---------------------------------------------------------------------------
# metric 5: metric residual (known ITF distances)
# ---------------------------------------------------------------------------
def classify_lines(lines):
    """Split painted lines into verticals {x_metre: idx} and horizontals
    {y_metre: idx} by which endpoint coordinate is constant."""
    verts, horiz = {}, {}
    for i, seg in enumerate(np.asarray(lines).reshape(-1, 2, 2)):
        dx = abs(seg[0, 0] - seg[1, 0])
        dy = abs(seg[0, 1] - seg[1, 1])
        if dx < dy:
            verts[round(float((seg[0, 0] + seg[1, 0]) / 2), 3)] = i
        else:
            horiz[round(float((seg[0, 1] + seg[1, 1]) / 2), 3)] = i
    return verts, horiz


def fit_frame_lines(white, polylines, size):
    """TLS image-line (a,b,c) for each painted line, from the white pixels in a
    band around its projection. None where too few pixels."""
    fitted = []
    for pl in polylines:
        if pl["valid"].sum() < 2:
            fitted.append(None)
            continue
        band = polyline_proximity_mask([pl], size, radius_px=int(METRIC_BAND))
        ys, xs = np.nonzero(white & band)
        fitted.append(fit_line_tls(np.column_stack([xs, ys])) if len(xs) >= 20 else None)
    return fitted


def metric_residual(Hc, model, white, polylines, size):
    """
    Median absolute error (cm) over the ITF distances that could be measured.
    Returns (median_cm, {name: error_cm}).
    """
    verts, horiz = classify_lines(model["lines"])
    fitted = fit_frame_lines(white, polylines, size)
    pts_m = model["pts"]

    def measure_landmark(k):
        x0, y0 = pts_m[k]
        vi = verts.get(round(float(x0), 3))
        hi = horiz.get(round(float(y0), 3))
        if vi is None or hi is None or fitted[vi] is None or fitted[hi] is None:
            return None
        px = intersect_lines(fitted[vi], fitted[hi])
        if px is None:
            return None
        m = project_points(Hc, px[None, :])[0]
        return m if np.isfinite(m).all() else None

    measured = {k: measure_landmark(k) for k in
                {i for a, b, _ in ITF_DISTANCES.values() for i in (a, b)}}
    errors = {}
    for name, (a, b, nominal) in ITF_DISTANCES.items():
        if measured.get(a) is None or measured.get(b) is None:
            continue
        d = float(np.linalg.norm(measured[a] - measured[b]))
        errors[name] = 100.0 * abs(d - nominal)
    med = float(np.median(list(errors.values()))) if errors else np.nan
    return med, errors


def reference_landmark_residual(ref_dir, results_dir, images, model):
    """
    Metric residual (cm) computed straight from the *raw* detector keypoints of
    the reference frame mapped through that frame's Hc - a direct read of the
    anchor's metric accuracy, independent of the per-frame line re-fitting.
    Returns (median_cm, {name: error_cm}) or (None, {}).
    """
    base_files = sorted(glob.glob(os.path.join(ref_dir, "court_base*.mat")))
    if not base_files or not images:
        return None, {}
    base = sio.loadmat(base_files[0])
    if "img_pts" not in base:
        return None, {}
    img_pts = np.asarray(base["img_pts"], dtype=np.float64)

    name = os.path.splitext(os.path.basename(images[0]))[0]
    num = frame_number(name)
    num_str = f"{num:04d}" if num is not None else name
    Hc = load_hc(results_dir, num_str)
    if Hc is None:
        return None, {}

    metres = project_points(Hc, img_pts)
    errors = {}
    for lname, (a, b, nominal) in ITF_DISTANCES.items():
        if not (np.isfinite(metres[a]).all() and np.isfinite(metres[b]).all()):
            continue
        d = float(np.linalg.norm(metres[a] - metres[b]))
        errors[lname] = 100.0 * abs(d - nominal)
    med = float(np.median(list(errors.values()))) if errors else None
    return med, errors


# ---------------------------------------------------------------------------
# metric 6: warped-court IoU
# ---------------------------------------------------------------------------
def polygon_iou(poly_a, poly_b, size, pad=IOU_PAD):
    w, h = int(round(size[0])), int(round(size[1]))
    canvas = (h + 2 * pad, w + 2 * pad)

    def raster(poly):
        m = np.zeros(canvas, dtype=np.uint8)
        q = np.round(np.asarray(poly) + pad).astype(np.int32)
        cv2.fillPoly(m, [q], 1)
        return m.astype(bool)

    A, B = raster(poly_a), raster(poly_b)
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    return float(inter / union) if union else np.nan


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def make_plots(rows, out_dir):
    nums = [r["frame"] for r in rows]

    def series(key):
        return [r[key] if r[key] is not None else np.nan for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(nums, series("line_median_px"), ".-", ms=3)
    axes[0].axhline(3.0, color="r", ls="--", lw=1, label="3 px target")
    axes[0].set_ylabel("line reproj (px)"); axes[0].legend(); axes[0].set_ylim(bottom=0)

    axes[1].plot(nums, series("metric_resid_cm"), ".-", ms=3, color="tab:green")
    axes[1].axhline(5.0, color="r", ls="--", lw=1, label="5 cm target")
    axes[1].set_ylabel("metric residual (cm)"); axes[1].legend(); axes[1].set_ylim(bottom=0)

    iou = series("iou")
    if np.any(np.isfinite(iou)):
        axes[2].plot(nums, iou, "o", ms=6, color="tab:purple")
        axes[2].axhline(0.95, color="r", ls="--", lw=1, label="0.95 target")
        axes[2].legend()
    axes[2].set_ylabel("warped-court IoU"); axes[2].set_xlabel("frame"); axes[2].set_ylim(0, 1)
    fig.suptitle("Court metrics (Appendix A: 1, 5, 6)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "court_metrics.png"), dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Court-based self-consistency metrics (Appendix A)")
    ap.add_argument("--results", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step-m", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = load_court_model(os.path.join(args.ref, "courtmodel.mat"))
    methods, inliers = load_methods(args.results)

    annot_path = os.path.join(args.ref, "iou_annotations.json")
    annotations = {}
    if os.path.isfile(annot_path):
        with open(annot_path) as f:
            annotations = json.load(f)
    else:
        print(f"[evaluate_court] no {annot_path}: IoU skipped "
              "(run tests/annotate_court.py to create it)")

    images = list_images(args.images)
    rows = []
    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        num = frame_number(name)
        num_str = f"{num:04d}" if num is not None else name
        Hc = load_hc(args.results, num_str)
        method = methods.get(num, "?")

        row = {"frame": num, "method": method, "inliers": inliers.get(num, 0),
               "line_median_px": None, "line_mean_px": None, "frac_inframe": None,
               "clean": False, "metric_resid_cm": None, "iou": None}

        if Hc is None:
            rows.append(row)
            continue

        img = cv2.imread(img_path)
        if img is None:
            rows.append(row)
            continue
        h, w = img.shape[:2]
        size = (w, h)
        H_court_to_img = np.linalg.inv(Hc)
        polylines = project_lines(H_court_to_img, model["lines"], size, step_m=args.step_m)

        white = white_line_mask(img)
        src = np.where(white, 0, 255).astype(np.uint8)
        dist_map = cv2.distanceTransform(src, cv2.DIST_L2, 5)

        frame_med, frame_mean, _, frac = line_reprojection_error(dist_map, polylines)
        clean = (method != "fallback") and (frac >= CLEAN_MIN_INFRAME)
        row.update(line_median_px=frame_med, line_mean_px=frame_mean,
                   frac_inframe=frac, clean=clean)

        med_cm, _ = metric_residual(Hc, model, white, polylines, size)
        row["metric_resid_cm"] = med_cm

        if num_str in annotations:
            warped_poly = project_points(H_court_to_img, model["polygon"])
            if np.isfinite(warped_poly).all():
                row["iou"] = polygon_iou(annotations[num_str], warped_poly, size)

        rows.append(row)

    # CSV
    csv_path = os.path.join(args.out, "per_frame.csv")
    fields = ["frame", "method", "inliers", "line_median_px", "line_mean_px",
              "frac_inframe", "clean", "metric_resid_cm", "iou"]
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    # summary over clean frames
    clean_rows = [r for r in rows if r["clean"]]

    def med(key, source):
        vals = [r[key] for r in source if r[key] is not None and np.isfinite(r[key])]
        return float(np.median(vals)) if vals else None

    iou_vals = [r["iou"] for r in rows if r["iou"] is not None and np.isfinite(r["iou"])]
    ref_resid, ref_errors = reference_landmark_residual(args.ref, args.results, images, model)
    summary = {
        "frames_total": len(rows),
        "frames_clean": len(clean_rows),
        "line_reproj_median_px_clean": med("line_median_px", clean_rows),
        "line_reproj_mean_px_clean": med("line_mean_px", clean_rows),
        "metric_residual_median_cm_clean": med("metric_resid_cm", clean_rows),
        "reference_landmark_residual_cm": ref_resid,
        "reference_landmark_residual_detail_cm": ref_errors,
        "iou_frames": len(iou_vals),
        "iou_median": float(np.median(iou_vals)) if iou_vals else None,
        "iou_min": float(np.min(iou_vals)) if iou_vals else None,
        "targets": {"line_reproj_px": "< 2-3", "metric_residual_cm": "< 5", "iou": "> 0.95"},
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    make_plots(rows, args.out)

    print(f"[evaluate_court] {len(rows)} frames, {len(clean_rows)} clean")
    print(f"  line reprojection (clean median) : {summary['line_reproj_median_px_clean']} px "
          f"(mean {summary['line_reproj_mean_px_clean']}) (target < 2-3)")
    print(f"  metric residual   (clean median) : {summary['metric_residual_median_cm_clean']} cm (target < 5)")
    print(f"  reference landmark residual      : {summary['reference_landmark_residual_cm']} cm (raw detector keypoints)")
    if iou_vals:
        print(f"  warped-court IoU  ({len(iou_vals)} frames) : median {summary['iou_median']:.3f}, "
              f"min {summary['iou_min']:.3f} (target > 0.95)")
    print(f"  -> {csv_path}")
    print(f"  -> {os.path.join(args.out, 'summary.json')}")
    print(f"  -> {os.path.join(args.out, 'court_metrics.png')}")


if __name__ == "__main__":
    main()
