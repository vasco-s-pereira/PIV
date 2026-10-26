#!/usr/bin/env python3
"""
Uso:
    python main1.py path_to_refdir path_images_dir path_feature_dir path_output_dir

O que faz:
  1. Extrai SIFT de refdir/templateimg.jpg e de todas as imagens de path_images_dir.
  2. Grava um .mat por imagem em path_feature_dir, no formato do pivist/features:
         {"kp": (2 + 128, N) float32}   linha 0 = x, linha 1 = y, linhas 2..129 = descritor
     As features do template ficam em path_feature_dir/templateimg.mat.
  3. Chama part1(refdir, images_dir, feature_dir, output_dir).

Extração: por omissão usa cv2.SIFT_create() com os parâmetros por omissão, que são
exatamente os do pivist/features e da caixa do VisionIST. Se o cv2 não existir,
usa a caixa do VisionIST no endereço PIV_FEATURES_BOX (por omissão localhost:9071).

Ficheiros de features já existentes e mais recentes que a imagem não são
recalculados (útil para correr o part1 várias vezes). Para forçar: PIV_FORCE_FEATURES=1.
"""
import os
import re
import sys

import numpy as np
from scipy.io import savemat

from part1 import part1

IMG_EXTS = (".jpg", ".jpeg", ".png")
TEMPLATE_STEM = "templateimg"
BOX_ADDR = os.environ.get("PIV_FEATURES_BOX", "localhost:9071")
FORCE = os.environ.get("PIV_FORCE_FEATURES", "0") == "1"
BATCH = 16

# Mesmos valores que o cv2.SIFT_create() usa por omissão (e que o enunciado indica).
SIFT_PARAMS = {
    "nfeatures": 0,
    "n_octave_layers": 3,
    "contrast_threshold": 0.04,
    "edge_threshold": 10,
    "sigma": 1.6,
    "draw": False,
    "mat": False,
}


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def frame_number(stem):
    """Número NNNN no fim do nome (somename_NNNN). None se não houver."""
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else None


def list_images(images_dir):
    """Imagens da sequência ordenadas pelo número de frame (não pela ordem alfabética)."""
    names = [f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS)]
    names.sort(key=lambda f: (frame_number(os.path.splitext(f)[0]) is None,
                              frame_number(os.path.splitext(f)[0]) or 0, f))
    return [os.path.join(images_dir, f) for f in names]


def find_template(refdir):
    for ext in IMG_EXTS:
        p = os.path.join(refdir, TEMPLATE_STEM + ext)
        if os.path.isfile(p):
            return p
    raise SystemExit(f"Não encontrei {TEMPLATE_STEM}.jpg em {refdir}")


def kp_matrix(points_xy, descriptors):
    """Monta a matriz (130, N) float32 do formato pivist/features. N pode ser 0."""
    if descriptors is None or len(points_xy) == 0:
        return np.zeros((130, 0), dtype=np.float32)
    xy = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2).T
    des = np.asarray(descriptors, dtype=np.float32).T
    return np.concatenate((xy, des), axis=0)


def is_up_to_date(img_path, mat_path):
    return (not FORCE and os.path.isfile(mat_path)
            and os.path.getmtime(mat_path) >= os.path.getmtime(img_path))


# ---------------------------------------------------------------------------
# Extratores
# ---------------------------------------------------------------------------
class Cv2Sift:
    def __init__(self):
        import cv2
        self.cv2 = cv2
        self.sift = cv2.SIFT_create()

    def __call__(self, img_path):
        im = self.cv2.imread(img_path)
        if im is None:
            raise IOError(f"não consegui ler {img_path}")
        # Igual ao pivist/features: converte para RGB antes do SIFT.
        imrgb = self.cv2.cvtColor(im, self.cv2.COLOR_BGR2RGB)
        kp, des = self.sift.detectAndCompute(imrgb, None)
        return kp_matrix([k.pt for k in kp], des)


def extract_with_box(jobs):
    """jobs = [(img_path, mat_path)]. Usa a caixa de features do VisionIST."""
    from visionist_client import Visionist
    import pathlib
    box = Visionist(BOX_ADDR)
    for start in range(0, len(jobs), BATCH):
        chunk = jobs[start:start + BATCH]
        paths = [pathlib.Path(p) for p, _ in chunk]
        res = box.run(
            data={"images": paths, "names": [p.stem for p in paths]},
            config={"features": {"command": "extract", "parameters": SIFT_PARAMS}},
        )
        kp = res.fields["keypoints"]          # (n, 130, N) com padding de zeros
        counts = res.fields["counts"]         # (n,) número real de keypoints
        for i, (img_path, mat_path) in enumerate(chunk):
            n = int(counts[i])
            savemat(mat_path, {"kp": np.asarray(kp[i, :, :n], dtype=np.float32)})
            print(f"  {os.path.basename(img_path)}: {n} keypoints")


def extract_features(jobs):
    todo = [(p, m) for p, m in jobs if not is_up_to_date(p, m)]
    print(f"[main1] features: {len(jobs) - len(todo)} em cache, {len(todo)} a calcular")
    if not todo:
        return
    try:
        sift = Cv2Sift()
    except ImportError:
        print(f"[main1] cv2 indisponível, a usar a caixa VisionIST em {BOX_ADDR}")
        extract_with_box(todo)
        return
    for k, (img_path, mat_path) in enumerate(todo, 1):
        kp = sift(img_path)
        savemat(mat_path, {"kp": kp})
        if k % 25 == 0 or k == len(todo):
            print(f"  {k}/{len(todo)}  {os.path.basename(img_path)}: {kp.shape[1]} keypoints")


# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    refdir, images_dir, feature_dir, output_dir = sys.argv[1:5]

    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    images = list_images(images_dir)
    if not images:
        raise SystemExit(f"Não há imagens em {images_dir}")

    template = find_template(refdir)
    jobs = [(template, os.path.join(feature_dir, TEMPLATE_STEM + ".mat"))]
    jobs += [(p, os.path.join(feature_dir, os.path.splitext(os.path.basename(p))[0] + ".mat"))
             for p in images]
    extract_features(jobs)

    part1(refdir, images_dir, feature_dir, output_dir)


if __name__ == "__main__":
    main()
