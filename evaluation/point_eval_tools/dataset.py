from __future__ import annotations

import os.path as osp
import re
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pycolmap
from PIL import Image
from torchvision import transforms as tvf


def natural_key(value: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def closed_form_inverse_se3(se3: np.ndarray) -> np.ndarray:
    rotation = se3[:3, :3]
    translation = se3[:3, 3]
    inverse = np.eye(4, dtype=np.float32)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def read_colmap_depth(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        width, height, channels = np.genfromtxt(
            handle, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        handle.seek(0)
        num_delimiters = 0
        byte = handle.read(1)
        while byte:
            if byte == b"&":
                num_delimiters += 1
                if num_delimiters >= 3:
                    break
            byte = handle.read(1)
        array = np.fromfile(handle, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def threshold_depth_map(
    depth_map: np.ndarray,
    max_percentile: float = 98,
    min_percentile: float = 2,
    max_depth: float = -1,
) -> np.ndarray:
    if depth_map is None:
        return None

    depth_map = depth_map.astype(np.float32, copy=True)
    if max_depth > 0:
        depth_map[depth_map > max_depth] = 0.0

    depth_max = np.nanpercentile(depth_map, max_percentile) if max_percentile > 0 else None
    depth_min = np.nanpercentile(depth_map, min_percentile) if min_percentile > 0 else None

    if depth_max is not None and depth_max > 0:
        depth_map[depth_map > depth_max] = 0.0
    if depth_min is not None and depth_min > 0:
        depth_map[depth_map < depth_min] = 0.0
    return depth_map.astype(np.float32)


def get_intrinsic_matrix(camera) -> np.ndarray:
    model_name = ""
    if hasattr(camera, "model_name"):
        model_name = camera.model_name
    elif hasattr(camera, "model"):
        model = camera.model
        model_name = model.name if hasattr(model, "name") else str(model)

    model_name = model_name.upper()
    intrinsics = np.eye(3, dtype=np.float32)

    if "SIMPLE" in model_name:
        intrinsics[0, 0] = camera.params[0]
        intrinsics[1, 1] = camera.params[0]
        intrinsics[0, 2] = camera.params[1]
        intrinsics[1, 2] = camera.params[2]
    elif "PINHOLE" in model_name:
        intrinsics[0, 0] = camera.params[0]
        intrinsics[1, 1] = camera.params[1]
        intrinsics[0, 2] = camera.params[2]
        intrinsics[1, 2] = camera.params[3]
    else:
        intrinsics[0, 0] = camera.params[0]
        intrinsics[1, 1] = camera.params[1] if len(camera.params) > 3 else camera.params[0]
        principal_index = 2 if len(camera.params) > 3 else 1
        intrinsics[0, 2] = camera.params[principal_index]
        intrinsics[1, 2] = camera.params[principal_index + 1]

    return intrinsics


def get_cam_from_world(image) -> np.ndarray:
    if hasattr(image, "cam_from_world"):
        transform = image.cam_from_world
        if callable(transform):
            transform = transform()
        if hasattr(transform, "matrix"):
            matrix = transform.matrix
            return matrix() if callable(matrix) else matrix

    if hasattr(image, "qvec") and hasattr(image, "tvec"):
        q = image.qvec
        t = image.tvec
        w, x, y, z = q[0], q[1], q[2], q[3]
        rotation = np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ],
            dtype=np.float32,
        )
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = t
        return extrinsic

    raise AttributeError(f"Could not extract extrinsics from image object: {image}")


def imread_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_max_round_min(
    image: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    target_size: int,
    multiple: int,
):
    image_pil = Image.fromarray(image) if not isinstance(image, Image.Image) else image
    width, height = image_pil.size

    if width >= height:
        scale = target_size / width
        new_width = int(target_size)
        new_height = max(multiple, int(round((height * scale) / multiple) * multiple))
    else:
        scale = target_size / height
        new_height = int(target_size)
        new_width = max(multiple, int(round((width * scale) / multiple) * multiple))

    scale_width = new_width / width
    scale_height = new_height / height
    resample = Image.Resampling.LANCZOS if scale_width < 1.0 and scale_height < 1.0 else Image.Resampling.BICUBIC
    image_resized = image_pil.resize((new_width, new_height), resample=resample)
    depth_resized = cv2.resize(depth_map, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    intrinsics = np.asarray(intrinsics, dtype=np.float32).copy()
    intrinsics[0, 0] *= scale_width
    intrinsics[1, 1] *= scale_height
    intrinsics[0, 2] *= scale_width
    intrinsics[1, 2] *= scale_height

    return image_resized, depth_resized.astype(np.float32), intrinsics


def depthmap_to_world_points(
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    cam_to_world: np.ndarray,
):
    height, width = depth_map.shape
    fu = intrinsics[0, 0]
    fv = intrinsics[1, 1]
    cu = intrinsics[0, 2]
    cv = intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_map
    x = (u - cu) * z / fu
    y = (v - cv) * z / fv
    cam_points = np.stack((x, y, z), axis=-1).astype(np.float32)

    rotation = cam_to_world[:3, :3]
    translation = cam_to_world[:3, 3]
    world_points = np.einsum("ij,hwj->hwi", rotation, cam_points) + translation[None, None, :]
    valid_mask = depth_map > 0.0
    return world_points.astype(np.float32), valid_mask


def uniform_sample_indices(num_items: int, num_samples: int) -> list[int]:
    if num_items <= 0 or num_samples <= 0:
        return []
    if num_items <= num_samples:
        return list(range(num_items))
    return np.round(np.linspace(0, num_items - 1, num_samples)).astype(int).tolist()


class Water3DManyViewDataset:
    def __init__(
        self,
        root: str | Path,
        num_views: int = 20,
        target_size: int = 518,
        multiple: int = 14,
        scenes: Iterable[str] | None = None,
        min_depth_percentile: float = 2,
        max_depth_percentile: float = 98,
        max_depth: float = -1,
    ):
        self.root = Path(root)
        self.num_views = int(num_views)
        self.target_size = int(target_size)
        self.multiple = int(multiple)
        self.scene_filter = set(scenes or [])
        self.min_depth_percentile = min_depth_percentile
        self.max_depth_percentile = max_depth_percentile
        self.max_depth = max_depth
        self.to_tensor = tvf.ToTensor()
        self.depth_suffixes = (".geometric.bin", ".photometric.bin", ".png")

        self.scenes: list[str] = []
        self.scene_frames: dict[str, list[dict]] = {}
        self._build_index()

    def __len__(self):
        return len(self.scenes)

    def _build_index(self):
        if not self.root.is_dir():
            raise FileNotFoundError(f"Water3D root not found: {self.root}")

        scene_dirs = sorted([path for path in self.root.iterdir() if path.is_dir()], key=lambda path: natural_key(path.name))
        for scene_root in scene_dirs:
            scene_name = scene_root.name
            if self.scene_filter and scene_name not in self.scene_filter:
                continue

            image_dir = scene_root / "images"
            if not image_dir.is_dir():
                image_dir = scene_root / "output" / "images"
            depth_dir = scene_root / "output" / "stereo" / "depth_maps"
            recon_dir = scene_root / "output" / "sparse"
            if not image_dir.is_dir() or not depth_dir.is_dir() or not recon_dir.is_dir():
                continue

            try:
                reconstruction = pycolmap.Reconstruction(str(recon_dir))
            except Exception as exc:
                print(f"Skipping {scene_name}: failed to read COLMAP reconstruction ({exc})")
                continue

            frames = []
            for image_id, image in reconstruction.images.items():
                image_name = osp.basename(image.name)
                if self._find_depth_path(depth_dir, image_name) is None:
                    continue
                if not (image_dir / image_name).is_file():
                    continue
                frames.append(
                    {
                        "scene": scene_name,
                        "image_name": image_name,
                        "image_dir": image_dir,
                        "depth_dir": depth_dir,
                        "recon_dir": recon_dir,
                        "image_id": image_id,
                    }
                )
            frames.sort(key=lambda item: natural_key(item["image_name"]))
            if frames:
                self.scenes.append(scene_name)
                self.scene_frames[scene_name] = frames

        if not self.scenes:
            raise RuntimeError(f"No valid Water3D scenes found under {self.root}")

    def _find_depth_path(self, depth_dir: Path, image_name: str) -> Path | None:
        for suffix in self.depth_suffixes:
            path = depth_dir / f"{image_name}{suffix}"
            if path.exists():
                return path
        return None

    def _load_frame(self, frame: dict) -> dict | None:
        image_name = frame["image_name"]
        depth_path = self._find_depth_path(frame["depth_dir"], image_name)
        if depth_path is None:
            return None

        depth_map = read_colmap_depth(depth_path)
        depth_map = threshold_depth_map(
            depth_map,
            min_percentile=self.min_depth_percentile,
            max_percentile=self.max_depth_percentile,
            max_depth=self.max_depth,
        )

        image = imread_rgb(frame["image_dir"] / image_name)
        if image.shape[:2] != depth_map.shape[:2]:
            image = cv2.resize(image, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_LINEAR)

        reconstruction = pycolmap.Reconstruction(str(frame["recon_dir"]))
        colmap_image = reconstruction.images[frame["image_id"]]
        camera = reconstruction.cameras[colmap_image.camera_id]
        intrinsics = get_intrinsic_matrix(camera)
        cam_from_world = get_cam_from_world(colmap_image).astype(np.float32)
        cam_to_world = closed_form_inverse_se3(cam_from_world)[:3, :].astype(np.float32)

        image, depth_map, intrinsics = resize_max_round_min(
            image,
            depth_map,
            intrinsics,
            target_size=self.target_size,
            multiple=self.multiple,
        )
        points_3d, valid_mask = depthmap_to_world_points(depth_map, intrinsics, cam_to_world)

        return {
            "img": self.to_tensor(image),
            "depthmap": depth_map,
            "camera_pose": cam_to_world,
            "camera_intrinsics": intrinsics,
            "pts3d": points_3d,
            "valid_mask": valid_mask & np.isfinite(points_3d).all(axis=-1),
            "dataset": "Water3D",
            "label": frame["scene"],
            "instance": image_name,
            "true_shape": np.int32((depth_map.shape[0], depth_map.shape[1])),
        }

    def __getitem__(self, index: int):
        scene = self.scenes[index]
        frames = self.scene_frames[scene]
        views = []
        for frame_index in uniform_sample_indices(len(frames), self.num_views):
            frame = self._load_frame(frames[frame_index])
            if frame is not None:
                views.append(frame)
        if not views:
            raise RuntimeError(f"No valid views found for scene {scene}")
        return views
