#!/usr/bin/env python3
"""
make_overlay_video.py - render the ITF court model over every frame (Appendix A:
"a short video with the court model overlaid on the original frames ... are worth
more than a page of prose - and they make failures impossible to hide").

For each frame it draws, using inv(Hc) (court metres -> frame pixels):
    * the painted court lines and the 14 landmarks,
    * a colour that encodes the tracking method from tracking_log.json
      (green = direct, yellow = sequential, red = fallback),
    * a text panel with the frame number, method, inlier count and - if a
      court_reports CSV is given - the line reprojection error,
    * optionally a top-view inset: the frame warped into court coordinates (Hc).

Every frame is written, including cuts/close-ups (drawn red via the propagated
last valid Hc), so the video covers the whole sequence like the outputs do.

Usage
-----
    uv run python tests/make_overlay_video.py \
        --results results/rally_04 \
        --images  datasets/rally_04 \
        --ref     ref/rally_04 \
        --out     court_reports/rally_04/overlay.mp4 \
        [--fps 25] [--reports court_reports/rally_04] [--minimap]
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main1 import list_images, frame_number                   # noqa: E402
from court_lines import load_court_model, project_lines, project_points   # noqa: E402

METHOD_COLOR = {                      # BGR
    "direct": (80, 220, 80),
    "sequential": (60, 220, 220),
    "fallback": (60, 60, 230),
    "?": (200, 200, 200),
}
# top-view minimap: metres -> inset pixels
MINIMAP_W, MINIMAP_H, MINIMAP_MARGIN = 220, 440, 16


def load_methods(results_dir):
    methods, inliers = {}, {}
    log = os.path.join(results_dir, "tracking_log.json")
    if os.path.isfile(log):
        with open(log) as f:
            for e in json.load(f):
                methods[e["frame"]] = e.get("method", "?")
                inliers[e["frame"]] = e.get("inliers", 0)
    return methods, inliers


def load_line_errors(reports_dir):
    errs = {}
    if not reports_dir:
        return errs
    path = os.path.join(reports_dir, "per_frame.csv")
    if os.path.isfile(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                v = row.get("line_median_px")
                try:
                    errs[int(row["frame"])] = float(v) if v not in (None, "", "None") else None
                except (ValueError, KeyError):
                    pass
    return errs


def draw_lines(img, polylines, color, thickness=2):
    for pl in polylines:
        xy, valid = pl["pts"], pl["valid"]
        run = []
        for i in range(len(xy)):
            if valid[i]:
                run.append(xy[i])
            else:
                if len(run) >= 2:
                    cv2.polylines(img, [np.round(run).astype(np.int32)], False, color, thickness, cv2.LINE_AA)
                run = []
        if len(run) >= 2:
            cv2.polylines(img, [np.round(run).astype(np.int32)], False, color, thickness, cv2.LINE_AA)


def court_to_minimap(pts_m, model):
    """Map court metres to minimap pixel coordinates."""
    xs = model["pts"][:, 0]; ys = model["pts"][:, 1]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    sx = (MINIMAP_W - 2 * MINIMAP_MARGIN) / (x1 - x0)
    sy = (MINIMAP_H - 2 * MINIMAP_MARGIN) / (y1 - y0)
    px = MINIMAP_MARGIN + (pts_m[:, 0] - x0) * sx
    py = MINIMAP_MARGIN + (y1 - pts_m[:, 1]) * sy       # y up -> down
    return np.column_stack([px, py])


def make_minimap(Hc, img, model):
    """Warp the frame into the metric court plane (top view) as a small inset."""
    inset = np.zeros((MINIMAP_H, MINIMAP_W, 3), dtype=np.uint8)
    # affine metres -> inset pixels
    corners_m = model["polygon"]
    dst = court_to_minimap(corners_m, model)
    src_px = project_points(np.linalg.inv(Hc), corners_m)
    if not np.isfinite(src_px).all():
        return inset
    M = cv2.getPerspectiveTransform(src_px.astype(np.float32), dst.astype(np.float32))
    warped = cv2.warpPerspective(img, M, (MINIMAP_W, MINIMAP_H))
    inset = warped
    lines_mm = [court_to_minimap(seg.reshape(2, 2), model) for seg in model["lines"]]
    for seg in lines_mm:
        cv2.line(inset, tuple(np.round(seg[0]).astype(int)), tuple(np.round(seg[1]).astype(int)),
                 (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(inset, (0, 0), (MINIMAP_W - 1, MINIMAP_H - 1), (200, 200, 200), 1)
    return inset


def main():
    ap = argparse.ArgumentParser(description="Overlay the court model on the video (Appendix A)")
    ap.add_argument("--results", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--reports", default=None, help="court_reports dir with per_frame.csv (line error)")
    ap.add_argument("--minimap", action="store_true", help="add a top-view inset")
    ap.add_argument("--step-m", type=float, default=0.05)
    args = ap.parse_args()

    model = load_court_model(os.path.join(args.ref, "courtmodel.mat"))
    methods, inliers = load_methods(args.results)
    line_errs = load_line_errors(args.reports)

    images = list_images(args.images)
    if not images:
        raise SystemExit(f"No images in {args.images}")

    first = cv2.imread(images[0])
    if first is None:
        raise SystemExit(f"Could not read {images[0]}")
    h, w = first.shape[:2]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"Could not open VideoWriter for {args.out}")

    n_written = 0
    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        num = frame_number(name)
        num_str = f"{num:04d}" if num is not None else name
        img = cv2.imread(img_path)
        if img is None:
            continue
        if (img.shape[1], img.shape[0]) != (w, h):
            img = cv2.resize(img, (w, h))

        method = methods.get(num, "?")
        color = METHOD_COLOR.get(method, METHOD_COLOR["?"])

        mat_path = os.path.join(args.results, f"homography_{num_str}.mat")
        Hc = None
        if os.path.isfile(mat_path):
            Hc = np.asarray(sio.loadmat(mat_path)["Hc"], dtype=np.float64).reshape(3, 3)
            if not np.all(np.isfinite(Hc)):
                Hc = None

        if Hc is not None:
            H_court_to_img = np.linalg.inv(Hc)
            polylines = project_lines(H_court_to_img, model["lines"], (w, h), step_m=args.step_m)
            draw_lines(img, polylines, color, 2)
            land = project_points(H_court_to_img, model["pts"])
            for p in land:
                if np.isfinite(p).all() and 0 <= p[0] < w and 0 <= p[1] < h:
                    cv2.circle(img, (int(p[0]), int(p[1])), 4, color, -1, cv2.LINE_AA)
            if args.minimap:
                inset = make_minimap(Hc, img.copy(), model)
                ih, iw = inset.shape[:2]
                img[h - ih - 10:h - 10, w - iw - 10:w - 10] = inset

        # text panel
        txt = f"frame {num_str}  {method}  inliers {inliers.get(num, 0)}"
        le = line_errs.get(num)
        if le is not None:
            txt += f"  line {le:.1f}px"
        cv2.rectangle(img, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(img, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        writer.write(img)
        n_written += 1

    writer.release()
    print(f"[make_overlay_video] wrote {n_written} frames at {args.fps} fps -> {args.out}")


if __name__ == "__main__":
    main()
