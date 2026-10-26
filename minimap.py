#!/usr/bin/env python3
"""
Minimap: where the players stood, in court metres. People are detected in every frame
(torchvision Faster R-CNN, MobileNet-v3 backbone, COCO "person"), the bottom-centre of
each box (the feet) is mapped to the court with Hc, and one player is kept per half:

    uv run python minimap.py datasets/rally_04 results/rally_04 ref/rally_04 videos/rally_04_minimap.mp4

Writes the video (original frame with the players boxed | top view of the court with
the players, their trails and the camera footprint) and <out>.players.json with the
per-frame positions. Detections are cached in <out>.detections.json, so re-rendering
is fast. Frames with no valid Hc (NaN, "no_court") show no positions.

Player selection, per half (near: y < 0, far: y > 0):
    * start on the most confident person with |x| <= START_X and |y| <= MAX_Y
      (line judges stand at |y| ~ 20 m, ball kids and the umpire at |x| ~ 7-9 m);
    * then follow the person closest to the last position, within MAX_STEP metres
      per frame, anywhere with |x| <= MAX_X and |y| <= MAX_Y;
    * lost for more than LOST_FRAMES frames -> start again.
"""
import argparse
import json
import os

import cv2
import numpy as np
import scipy.io as sio

from part1 import list_frames

SCORE_MIN = 0.5
START_X, MAX_X, MAX_Y = 5.5, 8.0, 17.0     # metres (court half-width 5.485, half-length 11.885)
MAX_STEP = 1.5                             # metres per frame (~45 m/s at 30 fps: generous)
LOST_FRAMES = 15
TRAIL = 45                                 # frames of trail on the minimap
PX_PER_M = 22                              # minimap scale
VIEW_X, VIEW_Y = 10.0, 19.0                # minimap extent (metres from the centre)


def detect_people(paths, cache_path):
    """{frame path: [[x0, y0, x1, y1, score], ...]}, cached in cache_path."""
    cache = json.load(open(cache_path)) if os.path.isfile(cache_path) else {}
    todo = [p for p in paths if os.path.basename(p) not in cache]
    if todo:
        import torch
        from torchvision.models import detection as D
        weights = D.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = D.fasterrcnn_mobilenet_v3_large_fpn(weights=weights).eval()
        for k, p in enumerate(todo, 1):
            img = cv2.imread(p)[:, :, ::-1].copy()
            x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            with torch.no_grad():
                o = model([x])[0]
            keep = (o["labels"] == 1) & (o["scores"] >= SCORE_MIN)
            boxes = torch.cat([o["boxes"][keep], o["scores"][keep, None]], dim=1)
            cache[os.path.basename(p)] = boxes.numpy().round(2).tolist()
            if k % 50 == 0 or k == len(todo):
                print(f"[minimap] detected {k}/{len(todo)}")
                json.dump(cache, open(cache_path, "w"))
    return {p: cache[os.path.basename(p)] for p in paths}


def to_court(Hc, pts):
    p = np.column_stack([pts, np.ones(len(pts))]) @ Hc.T
    return p[:, :2] / p[:, 2:3]


class PlayerTrack:
    """One player in one half of the court (sign = -1 near, +1 far)."""

    def __init__(self, sign):
        self.sign, self.pos, self.missed = sign, None, LOST_FRAMES + 1

    def update(self, court_xy, scores):
        """Pick this half's player among the detections; returns its index or None."""
        in_half = (np.sign(court_xy[:, 1]) == self.sign) & (np.abs(court_xy[:, 1]) <= MAX_Y)
        best = None
        if self.pos is not None and self.missed <= LOST_FRAMES:
            ok = in_half & (np.abs(court_xy[:, 0]) <= MAX_X)
            d = np.linalg.norm(court_xy - self.pos, axis=1)
            ok &= d <= MAX_STEP * (self.missed + 1)
            if ok.any():
                best = int(np.argmin(np.where(ok, d, np.inf)))
        else:
            ok = in_half & (np.abs(court_xy[:, 0]) <= START_X)
            if ok.any():
                best = int(np.argmax(np.where(ok, scores, -1)))
        if best is None:
            self.missed += 1
        else:
            self.pos, self.missed = court_xy[best], 0
        return best

    def reset(self):
        self.pos, self.missed = None, LOST_FRAMES + 1


def court_canvas(lines):
    W, H = int(2 * VIEW_X * PX_PER_M), int(2 * VIEW_Y * PX_PER_M)
    img = np.full((H, W, 3), (60, 110, 60), np.uint8)
    for p, q in lines:
        cv2.line(img, m2px(p), m2px(q), (255, 255, 255), 2, cv2.LINE_AA)
    return img


def m2px(xy):
    return (int(round((xy[0] + VIEW_X) * PX_PER_M)), int(round((VIEW_Y - xy[1]) * PX_PER_M)))


def footprint(Hc, w, h):
    """Camera footprint on the court (frame corners mapped with Hc), or None."""
    c = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64) @ Hc.T
    if not np.all(c[:, 2] > 0):
        return None
    c = c[:, :2] / c[:, 2:3]
    c = np.clip(c, [-3 * VIEW_X, -3 * VIEW_Y], [3 * VIEW_X, 3 * VIEW_Y])
    return np.array([m2px(p) for p in c], np.int32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images_dir")
    ap.add_argument("results_dir", help="part1 output (homography_NNNN.mat, tracking_log.json)")
    ap.add_argument("refdir", help="reference directory with courtmodel.mat (court lines)")
    ap.add_argument("out", help="output .mp4")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    frames = list_frames(args.images_dir)
    paths = [next(os.path.join(args.images_dir, n + e) for e in (".jpg", ".jpeg", ".png")
                  if os.path.isfile(os.path.join(args.images_dir, n + e))) for _, _, n in frames]
    log_path = os.path.join(args.results_dir, "tracking_log.json")
    log = {e["frame"]: e for e in json.load(open(log_path))} if os.path.isfile(log_path) else {}
    lines = sio.loadmat(os.path.join(args.refdir, "courtmodel.mat"))["lines"]

    stem = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dets = detect_people(paths, stem + ".detections.json")

    base = court_canvas(lines)
    mh, mw = base.shape[:2]
    h, w = cv2.imread(paths[0]).shape[:2]
    fw = int(round(mh * w / h))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (fw + mw, mh))
    if not writer.isOpened():
        raise SystemExit(f"Cannot write {args.out}")

    tracks = {"near": PlayerTrack(-1), "far": PlayerTrack(+1)}
    colours = {"near": (255, 80, 0), "far": (0, 140, 255)}
    trails = {k: [] for k in tracks}
    positions = []

    for (num, num_str, _), path in zip(frames, paths):
        frame = cv2.imread(path)
        Hc = sio.loadmat(os.path.join(args.results_dir, f"homography_{num_str}.mat"))["Hc"]
        mini = base.copy()
        row = {"frame": num}
        boxes = np.array(dets[path], dtype=np.float64).reshape(-1, 5)

        if np.all(np.isfinite(Hc)) and len(boxes):
            fp = footprint(Hc, w, h)
            if fp is not None:
                overlay = mini.copy()
                cv2.fillPoly(overlay, [fp], (200, 200, 200))
                mini = cv2.addWeighted(overlay, 0.18, mini, 0.82, 0)
            feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]])
            court_xy = to_court(Hc, feet)
            for b in boxes:
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (160, 160, 160), 1)
            for k, t in tracks.items():
                i = t.update(court_xy, boxes[:, 4])
                if i is not None:
                    row[k] = [round(float(v), 3) for v in court_xy[i]]
                    trails[k].append(court_xy[i])
                    b = boxes[i]
                    cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), colours[k], 3)
        else:
            for k, t in tracks.items():
                t.reset()
                trails[k].clear()
            text = f"no court ({log.get(num, {}).get('reason', 'no valid Hc')})"
            cv2.putText(mini, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        for k in tracks:
            trails[k] = trails[k][-TRAIL:]
            pts = [m2px(p) for p in trails[k]]
            for a, b in zip(pts, pts[1:]):
                cv2.line(mini, a, b, colours[k], 2, cv2.LINE_AA)
            if k in row:
                cv2.circle(mini, m2px(row[k]), 8, colours[k], -1, cv2.LINE_AA)
                cv2.circle(mini, m2px(row[k]), 8, (255, 255, 255), 2, cv2.LINE_AA)
        positions.append(row)

        cv2.putText(frame, os.path.basename(path), (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(np.hstack([cv2.resize(frame, (fw, mh)), mini]))
    writer.release()

    json.dump(positions, open(stem + ".players.json", "w"))
    seen = {k: sum(k in r for r in positions) for k in tracks}
    print(f"[minimap] {len(positions)} frames, players found: near {seen['near']}, far {seen['far']} "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
