from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "SeathruNeRF"


def _image_number(name):
    match = re.search(r"\d+", name)
    return int(match.group()) if match else -1


def _read_colmap_images(images_txt, image_root, random_order=False, seed=0):
    lines = [
        line.strip()
        for line in Path(images_txt).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    frames = []
    for line in lines[::2]:
        values = line.split()
        if len(values) < 10:
            continue
        qw, qx, qy, qz = map(float, values[1:5])
        tx, ty, tz = map(float, values[5:8])
        image_name = values[9]
        image_path = image_root / image_name
        if not image_path.is_file():
            continue
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        extrinsic[:3, 3] = [tx, ty, tz]
        frames.append((image_path, extrinsic))

    frames.sort(key=lambda item: _image_number(item[0].name))
    if random_order:
        random.Random(seed).shuffle(frames)
    return frames


def load_seathru_nerf(dataset_root=None, random_order=False, seed=0):
    root = Path(dataset_root) if dataset_root else DEFAULT_DATASET_ROOT
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SeaThru-NeRF directory does not exist: {root}")

    sequences = {}
    for scene_root in sorted(path for path in root.iterdir() if path.is_dir()):
        image_root = scene_root / "Images_wb"
        if not image_root.is_dir():
            image_root = scene_root / "images_wb"
        images_txt = scene_root / "sparse" / "1" / "images.txt"
        if image_root.is_dir() and images_txt.is_file():
            frames = _read_colmap_images(images_txt, image_root, random_order, seed)
            if len(frames) >= 2:
                sequences[scene_root.name] = frames
    if not sequences:
        raise RuntimeError(f"No valid SeaThru-NeRF sequences found under {root}")
    return sequences
