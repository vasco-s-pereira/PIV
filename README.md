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
For each rally you want to evaluate, you need to extract the first frame (as `templateimg.jpg`) and compute the initial 14 court keypoints (`court_base_templateimg.mat`).

1. Extract the first frame of the rally and save it. For instance, save it as `templateimg.jpg`.
2. Run the `TennisCourtDetector` inference on that image to extract the keypoints:
   ```bash
   cd TennisCourtDetector
   uv run --project .. python infer_in_image.py \
     --model_path models/model.pt \
     --input_path ../templateimg.jpg \
     --output_path ../templateimg_kps.jpg \
     --use_homography --use_refine_kps
   ```
3. This writes an annotated image. Save the 14 keypoints as `court_base_templateimg.mat` with an `img_pts` array of shape `(N, 2)` (pixel positions in `templateimg.jpg`). 
4. Place **both** `templateimg.jpg` and `court_base_templateimg.mat` into your reference directory (e.g., `ref_tennis/` or a rally-specific directory). The evaluation scripts will look for `court_base_*.mat` in either the `images_dir` or the `ref_dir`.

> **Note:** The universal physical dimensions of the ITF tennis court are provided in `ref_tennis/courtmodel.mat`. This file is already tracked in the repository and you do not need to generate it.

## Running the Tracking Pipeline

Once the data is prepared, run the evaluation script inside the environment (`uv run`, or after `source .venv/bin/activate`):

```bash
uv run python main1.py <path_to_refdir> <path_images_dir> <path_feature_dir> <path_output_dir>
```

Example:
```bash
uv run python main1.py ref_tennis/ datasets/rally_04/images/ features/rally_04/ results/rally_04/
```
