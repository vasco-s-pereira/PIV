# Tennis Court Tracking (PIV Project)

This repository contains the implementation for tracking tennis court lines across a video sequence using homography estimation, part of the PIV course project.

## Requirements

Dependencies are managed with [uv](https://docs.astral.sh/uv/). From this directory:

```bash
uv sync
```

That creates a `.venv` and installs NumPy, SciPy, OpenCV, Matplotlib, and PyTorch (CPU) from `pyproject.toml` / `uv.lock`.

## Setup & Data Preparation

Due to the size of the model weights and datasets, they are not included in this repository. You must set them up manually before running the tracking algorithms.

### 1. Download the Dataset
You can download the rally video datasets and their associated data from this [Google Drive folder](https://drive.google.com/drive/folders/1_FGsEY-lHNNxEBOCPWyPXaoLSU2AD__X?usp=sharing). Once downloaded, place the rallies in your desired local directory (e.g., `datasets/rally_04/`, etc.).

### 2. Download and Setup TennisCourtDetector
This project relies on the [TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector) to extract the initial reference court keypoints (`court_base_<name>.mat`) from the first frame of the rally.

1. Clone the `TennisCourtDetector` repository inside this project directory (or elsewhere):
   ```bash
   git clone https://github.com/yastrebksv/TennisCourtDetector.git TennisCourtDetector
   ```
2. PyTorch is already installed by `uv sync` above. Do not run `pip install -r TennisCourtDetector/requirements.txt`: that file pins Torch 1.5 and NumPy 1.19, and the detector does not use `fastai`.
3. Download the [pretrained weights](https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view) and save them as `TennisCourtDetector/models/model.pt` (this folder is gitignored).

### 3. Extract Reference Images and Keypoints
For each rally, copy its first frame to `templateimg.jpg` and detect the 14 court keypoints. `extract_reference.py` does both and writes `court_base_templateimg.mat` (`img_pts`, shape `(14, 2)`, same landmark order as `courtmodel.mat`):

```bash
uv run python extract_reference.py datasets/rally_04 ref/rally_04
```

Use one reference directory per rally (`ref/rally_01`, ..., `ref/rally_04`): the keypoints belong to that rally's first frame. That also writes `templateimg_kps.jpg` so you can check the points. If `refdir` has no `courtmodel.mat`, the script links the one in `ref_tennis/`. `part1` looks for `court_base_*.mat` in the reference directory first, then in the images directory, and uses the first one found.

> **Note:** The universal physical dimensions of the ITF tennis court are provided in `ref_tennis/courtmodel.mat`. This file is already tracked in the repository and you do not need to generate it.

## Running the Tracking Pipeline

Once the data is prepared, run the evaluation script inside the environment (`uv run`, or after `source .venv/bin/activate`):

```bash
uv run python main1.py <path_to_refdir> <path_images_dir> <path_feature_dir> <path_output_dir>
```

Frames sit directly in `datasets/rally_XX/`, and each rally has its own reference under `ref/rally_XX/`. For every rally:

```bash
uv run python main1.py ref/rally_04 datasets/rally_04 features/rally_04 results/rally_04
```

All rallies:
```bash
for r in rally_01 rally_02 rally_03 rally_04; do
  uv run python extract_reference.py datasets/$r ref/$r
  uv run python main1.py ref/$r datasets/$r features/$r results/$r
done
```

`extract_reference.py` now also saves, in `court_base_templateimg.mat`, the raw
detector keypoints (`img_pts`), the detector-snapped points (`img_pts_snapped`),
the white court-line pixels (`line_px`) and the image `size`. The anchor in
`part1.py` fits the metric court homography in the court-to-image direction with
an exhaustive 4-point RANSAC over the landmarks (which rejects a mislocated
keypoint), LM refinement, and a final robust snap onto `line_px`; it writes
`anchor_report.json` to the output directory. If no court landmark file is found,
`Hc` is written as all-`NaN` (an explicitly invalid metric homography is safer
than a silently wrong one), while `H` is still produced for every frame.

## Court Model Metrics and Overlay Video (Appendix A)

These scripts evaluate and visualise the metric court anchor `Hc`.

Hand-mark the doubles-court corners on a few frames (independent ground truth for
the warped-court IoU; uses a matplotlib window):
```bash
uv run python tests/annotate_court.py --images datasets/rally_04 --ref ref/rally_04 --n 5
```

Compute the court metrics (line reprojection error, metric residual in cm, and
warped-court IoU where annotations exist); writes a per-frame CSV, `summary.json`
and plots:
```bash
uv run python tests/evaluate_court.py \
  --results results/rally_04 --images datasets/rally_04 \
  --ref ref/rally_04 --out court_reports/rally_04
```

Render the ITF court model over every frame (green = direct, yellow = sequential,
red = fallback), with an optional top-view minimap inset:
```bash
uv run python tests/make_overlay_video.py \
  --results results/rally_04 --images datasets/rally_04 --ref ref/rally_04 \
  --out court_reports/rally_04/overlay.mp4 --reports court_reports/rally_04 --minimap
```

Unit tests for the anchor (synthetic RANSAC/refinement, no-cv2 check):
```bash
uv run python tests/test_court.py
```
