#!/usr/bin/env python3
"""
annotate_court.py - hand-mark the doubles-court outline on a few frames, to give
the warped-court IoU metric (Appendix A, metric 6) an independent ground truth.

Appendix A says explicitly: "mark the court polygon by hand and compare with the
warped model polygon". This tool opens a handful of frames and lets you click the
four doubles corners; the clicks are saved and read back by evaluate_court.py.

It uses matplotlib for the UI (the project ships opencv-python-headless, which has
no cv2.imshow). The axes are expanded by --pad px beyond the image on every side,
so corners that fall outside the frame can still be clicked; coordinates are saved
in the original image frame (negative / out-of-range values are fine).

Usage
-----
    uv run python tests/annotate_court.py \
        --images datasets/rally_04 \
        --ref ref/rally_04 \
        [--results results/rally_04] [--frames 0,40,80] [--n 5] [--pad 200]

Controls (matplotlib ginput, one frame at a time)
    left click   : drop a corner, in the order  TL, TR, BR, BL  (doubles corners)
    right click  : undo the last click
    enter        : finish this frame (needs exactly 4 clicks; otherwise it is skipped)

Output: ref/rally_XX/iou_annotations.json
    {"0040": [[x, y], [x, y], [x, y], [x, y]], ...}   # TL, TR, BR, BL
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main1 import list_images, frame_number   # noqa: E402

CORNER_NAMES = ["TL (doubles)", "TR (doubles)", "BR (doubles)", "BL (doubles)"]


def choose_frames(images, args):
    if args.frames:
        want = [int(x) for x in args.frames.split(",") if x.strip() != ""]
        chosen = []
        for w in want:
            match = min(images, key=lambda p: abs(
                (frame_number(os.path.splitext(os.path.basename(p))[0]) or 0) - w))
            chosen.append(match)
        return chosen
    methods = {}
    if args.results:
        log = os.path.join(args.results, "tracking_log.json")
        if os.path.isfile(log):
            with open(log) as f:
                for e in json.load(f):
                    methods[e["frame"]] = e["method"]
    good = [i for i in range(len(images))
            if methods.get(frame_number(os.path.splitext(os.path.basename(images[i]))[0]),
                           "direct") != "fallback"]
    pool = good if len(good) >= args.n else list(range(len(images)))
    picks = np.linspace(0, len(pool) - 1, min(args.n, len(pool))).round().astype(int)
    return [images[pool[p]] for p in picks]


def annotate_one(img_path, pad):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[annotate] could not read {img_path}")
        return None
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(rgb)
    ax.set_xlim(-pad, w + pad)
    ax.set_ylim(h + pad, -pad)                       # image y grows downward
    ax.set_title(f"{os.path.basename(img_path)}\n"
                 "click TL, TR, BR, BL  |  right-click = undo  |  enter = done",
                 fontsize=10)
    ax.set_facecolor((0.12, 0.12, 0.12))

    pts = plt.ginput(n=4, timeout=0, show_clicks=True,
                     mouse_add=1, mouse_pop=3, mouse_stop=2)
    plt.close(fig)

    if len(pts) != 4:
        return "skip"
    return [[float(x), float(y)] for x, y in pts]


def main():
    ap = argparse.ArgumentParser(description="Hand-mark doubles-court corners for IoU ground truth")
    ap.add_argument("--images", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--results", default=None)
    ap.add_argument("--frames", default=None, help="comma list of frame numbers, e.g. 0,40,80")
    ap.add_argument("--n", type=int, default=5, help="how many frames when --frames is omitted")
    ap.add_argument("--pad", type=int, default=200)
    args = ap.parse_args()

    images = list_images(args.images)
    if not images:
        raise SystemExit(f"No images in {args.images}")
    frames = choose_frames(images, args)

    os.makedirs(args.ref, exist_ok=True)
    out_path = os.path.join(args.ref, "iou_annotations.json")
    data = {}
    if os.path.isfile(out_path):
        with open(out_path) as f:
            data = json.load(f)

    for img_path in frames:
        num = frame_number(os.path.splitext(os.path.basename(img_path))[0])
        key = f"{num:04d}"
        print(f"[annotate] frame {key}  ({os.path.basename(img_path)}) - click 4 corners, enter to confirm")
        res = annotate_one(img_path, args.pad)
        if res == "skip" or res is None:
            print("           skipped (need exactly 4 clicks)")
            continue
        data[key] = res
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print("           saved 4 corners")

    print(f"[annotate] {len(data)} frame(s) annotated -> {out_path}")


if __name__ == "__main__":
    main()
