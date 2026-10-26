#!/usr/bin/env python3
"""
Stabilised video: every frame warped into the reference image (frame 0) with the H
that part1 wrote, so the court stays still while the camera pans and zooms.

    uv run python stabilise_video.py datasets/rally_04 results/rally_04 videos/rally_04_stabilised.mp4

The canvas is the reference image enlarged to the bounding box of all warped frames
(capped at --max-scale times the reference), so the whole pan fits. Frames with no
valid H (NaN, method "no_court" in tracking_log.json) are shown dimmed with the reason.

    --mosaic       keep previously warped pixels as background (a growing panorama)
    --side-by-side original frame on the left, stabilised on the right
"""
import argparse
import json
import os

import cv2
import numpy as np
import scipy.io as sio

from part1 import list_frames


def load_track(images_dir, results_dir):
    """[(frame path, H or None, log entry)] in frame order."""
    log = {}
    log_path = os.path.join(results_dir, "tracking_log.json")
    if os.path.isfile(log_path):
        log = {e["frame"]: e for e in json.load(open(log_path))}
    track = []
    for num, num_str, name in list_frames(images_dir):
        img = next(os.path.join(images_dir, name + ext) for ext in (".jpg", ".jpeg", ".png")
                   if os.path.isfile(os.path.join(images_dir, name + ext)))
        H = sio.loadmat(os.path.join(results_dir, f"homography_{num_str}.mat"))["H"]
        track.append((img, H if np.all(np.isfinite(H)) else None, log.get(num, {})))
    return track


def canvas_transform(track, w, h, max_scale):
    """Translation T (reference -> canvas) and canvas size covering all warped frames."""
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64)
    pts = [corners[:, :2]]
    for _, H, _ in track:
        if H is not None:
            p = corners @ H.T
            if np.all(p[:, 2] > 0):
                pts.append(p[:, :2] / p[:, 2:3])
    pts = np.vstack(pts)
    lo = np.maximum(pts.min(axis=0), [-(max_scale - 1) / 2 * w, -(max_scale - 1) / 2 * h])
    hi = np.minimum(pts.max(axis=0), [(max_scale + 1) / 2 * w, (max_scale + 1) / 2 * h])
    T = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]], dtype=np.float64)
    return T, (int(np.ceil(hi[0] - lo[0])), int(np.ceil(hi[1] - lo[1])))


def label(img, text, colour=(0, 255, 255), line=0):
    y = 45 + 45 * line
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images_dir")
    ap.add_argument("results_dir", help="part1 output (homography_NNNN.mat, tracking_log.json)")
    ap.add_argument("out", help="output .mp4")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-scale", type=float, default=3.0, help="canvas size cap, in reference sizes")
    ap.add_argument("--max-width", type=int, default=1920, help="downscale the output to this width")
    ap.add_argument("--mosaic", action="store_true")
    ap.add_argument("--side-by-side", action="store_true")
    args = ap.parse_args()

    track = load_track(args.images_dir, args.results_dir)
    if not track:
        raise SystemExit(f"No frames in {args.images_dir}")
    h, w = cv2.imread(track[0][0]).shape[:2]
    T, (cw, ch) = canvas_transform(track, w, h, args.max_scale)
    side = ch * w / h if args.side_by_side else 0      # original frame, scaled to canvas height
    s = min(1.0, args.max_width / (cw + side))
    S = np.diag([s, s, 1.0]) @ T
    cw, ch = int(round(cw * s)), int(round(ch * s))
    side_w = int(round(ch * w / h)) if args.side_by_side else 0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (side_w + cw, ch))
    if not writer.isOpened():
        raise SystemExit(f"Cannot write {args.out}")

    canvas = np.zeros((ch, cw, 3), np.uint8)
    # frame mask shrunk by a few pixels: border pixels interpolate with black and streak the mosaic
    inner = np.zeros((h, w), np.uint8)
    inner[3:-3, 3:-3] = 255
    n_nan = 0
    for img_path, H, entry in track:
        frame = cv2.imread(img_path)
        if not args.mosaic:
            canvas[:] = 0
        if H is not None:
            warped = cv2.warpPerspective(frame, S @ H, (cw, ch))
            mask = cv2.warpPerspective(inner, S @ H, (cw, ch), flags=cv2.INTER_NEAREST) > 0
            canvas[mask] = warped[mask]
            out = canvas.copy()
        else:
            n_nan += 1
            out = (canvas * 0.3).astype(np.uint8)
            label(out, f"no court ({entry.get('reason', 'no valid H')})", (0, 0, 255), line=1)
        label(out, f"{os.path.basename(img_path)}  {entry.get('method', '')}")
        if args.side_by_side:
            out = np.hstack([cv2.resize(frame, (side_w, ch)), out])
        writer.write(out)
    writer.release()
    print(f"[stabilise] {len(track)} frames ({n_nan} without a valid H), "
          f"canvas {cw}x{ch} -> {args.out}")


if __name__ == "__main__":
    main()
