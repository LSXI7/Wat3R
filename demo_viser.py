# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Visualize Wat3R reconstructions with Viser.

The demo supports an accumulated reconstruction, per-frame playback, and
static/dynamic playback based on multi-view depth consistency.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import torch
import viser
import viser.transforms as viser_tf

from wat3r.models.wat3r import Wat3R
from wat3r.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from wat3r.utils.load_fn import load_and_preprocess_images
from wat3r.utils.pose_enc import pose_encoding_to_extri_intri
from wat3r.utils.static_mask import build_static_masks, depth_foreground_mask


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wat3R 3D visualization with Viser")
    parser.add_argument("--input", type=Path, required=True, help="Image file or image directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Wat3R checkpoint")
    parser.add_argument(
        "--mode",
        choices=("static", "playback", "dynamic"),
        default="static",
        help="Accumulated scene, per-frame playback, or static/dynamic playback",
    )
    parser.add_argument(
        "--use_point_map",
        action="store_true",
        help="Visualize the point-head output instead of unprojected depth",
    )
    parser.add_argument(
        "--foreground_only",
        action="store_true",
        help="Restrict dynamic-mode static-mask candidates to the near K-means depth cluster",
    )
    parser.add_argument(
        "--visibility_tolerance",
        type=int,
        default=2,
        help="Frames in which a static point may be inconsistent (default: 2)",
    )
    parser.add_argument("--teacher", action="store_true", help="Use ema_models from a training checkpoint")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=25.0,
        help="Initial percentage of low-confidence points to remove",
    )
    parser.add_argument(
        "--sort_input",
        choices=("none", "name", "numeric"),
        default="numeric",
        help="Input image ordering",
    )
    parser.add_argument(
        "--save",
        "--save_predictions",
        dest="save_predictions",
        type=Path,
        nargs="?",
        const=Path("predictions.pt"),
        help="Save CPU predictions, optionally to the given path",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device",
    )
    args = parser.parse_args()

    if not 0 <= args.conf_threshold <= 100:
        parser.error("--conf_threshold must be in [0, 100]")
    if args.visibility_tolerance < 0:
        parser.error("--visibility_tolerance must be non-negative")
    if args.foreground_only and args.mode != "dynamic":
        parser.error("--foreground_only is only valid with --mode dynamic")
    return args


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def natural_key(path: Path) -> tuple:
    return tuple(
        (0, int(token)) if token.isdigit() else (1, token.lower())
        for token in re.split(r"(\d+)", path.name)
        if token
    )


def collect_images(input_path: Path, sorting: str) -> list[Path]:
    if input_path.is_file():
        images = [input_path]
    elif input_path.is_dir():
        images = [path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not images:
        raise ValueError(f"No supported images found in {input_path}")
    if sorting == "name":
        images.sort(key=lambda path: path.name.lower())
    elif sorting == "numeric":
        images.sort(key=natural_key)
    return images


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    *,
    teacher: bool,
    need_depth: bool,
    need_point: bool,
) -> Wat3R:
    model = Wat3R(
        enable_track=False,
        enable_point=need_point,
        enable_depth=need_depth,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if teacher:
        if not isinstance(checkpoint, dict) or "ema_models" not in checkpoint:
            raise KeyError("--teacher requires an ema_models entry in the checkpoint")
        state_dict = checkpoint["ema_models"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    ignored_unexpected = ["track_head."]
    if not need_depth:
        ignored_unexpected.append("depth_head.")
    if not need_point:
        ignored_unexpected.append("point_head.")
    unexpected = [key for key in unexpected if not key.startswith(tuple(ignored_unexpected))]
    if missing:
        print(f"Missing checkpoint keys: {missing}")
    if unexpected:
        print(f"Unexpected checkpoint keys: {unexpected}")
    return model.to(device).eval()


@torch.no_grad()
def run_inference(
    model: Wat3R,
    image_paths: list[Path],
    device: torch.device,
    *,
    need_depth: bool,
    need_point: bool,
) -> dict[str, torch.Tensor]:
    images = load_and_preprocess_images(
        [str(path) for path in image_paths], mode="max", target_size=518
    ).to(device)
    amp_dtype = torch.float32
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
        predictions = model(
            images,
            need_depth=need_depth,
            need_point=need_point,
            need_camera=True,
        )
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsics
    predictions["intrinsic"] = intrinsics
    return predictions


def prepare_dynamic_mask(
    predictions: dict[str, torch.Tensor],
    *,
    foreground_only: bool,
    visibility_tolerance: int,
) -> None:
    depth = predictions["depth"].squeeze(-1)
    candidates = depth_foreground_mask(depth) if foreground_only else None
    predictions["static_mask"] = build_static_masks(
        depth,
        predictions["extrinsic"],
        predictions["intrinsic"],
        candidate_masks=candidates,
        visibility_tolerance=visibility_tolerance,
    )


def move_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: move_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_cpu(item) for item in value)
    return value


def cpu_predictions(predictions: dict) -> dict:
    return move_to_cpu(predictions)


def move_to_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
        return value[0] if value.ndim > 0 and value.shape[0] == 1 else value
    if isinstance(value, dict):
        return {key: move_to_numpy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_numpy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_numpy(item) for item in value)
    return value


def numpy_predictions(predictions: dict) -> dict:
    return move_to_numpy(predictions)


class ViserDemo:
    def __init__(
        self,
        predictions: dict,
        *,
        mode: str,
        use_point_map: bool,
        port: int,
        conf_threshold: float,
    ) -> None:
        self.mode = mode
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        self.images = predictions["images"]
        self.extrinsics = predictions["extrinsic"]
        self.intrinsics = predictions["intrinsic"]
        if use_point_map:
            self.world_points = predictions["world_points"]
            self.confidence = predictions["world_points_conf"]
        else:
            self.world_points = unproject_depth_map_to_point_map(
                predictions["depth"], self.extrinsics, self.intrinsics
            )
            self.confidence = predictions["depth_conf"]

        self.static_masks = predictions.get("static_mask")
        self.num_frames, self.height, self.width = self.confidence.shape
        self.points_by_frame = self.world_points.reshape(self.num_frames, -1, 3)
        self.colors_by_frame = (
            self.images.transpose(0, 2, 3, 1).reshape(self.num_frames, -1, 3) * 255
        ).clip(0, 255).astype(np.uint8)
        self.confidence_by_frame = self.confidence.reshape(self.num_frames, -1)
        self.static_masks_by_frame = (
            None if self.static_masks is None else self.static_masks.reshape(self.num_frames, -1).astype(bool)
        )

        finite_points = np.isfinite(self.points_by_frame).all(axis=-1)
        if not finite_points.any():
            raise ValueError("The reconstruction contains no finite 3D points")
        self.scene_center = self.points_by_frame[finite_points].mean(axis=0)
        self.points_by_frame = self.points_by_frame - self.scene_center
        self.initial_conf_threshold = conf_threshold

        self._build_gui()
        self._build_point_clouds()
        self._build_cameras()
        self.update_point_clouds()

    def _build_gui(self) -> None:
        self.show_cameras = self.server.gui.add_checkbox("Show Cameras", initial_value=True)
        self.conf_slider = self.server.gui.add_slider(
            "Confidence Percent",
            min=0.0,
            max=100.0,
            step=0.1,
            initial_value=self.initial_conf_threshold,
        )
        self.frame_selector = None
        self.timestep = None
        self.playing = None
        self.fps = None

        if self.mode == "static":
            self.frame_selector = self.server.gui.add_dropdown(
                "Show Points from Frames",
                options=["All"] + [str(index) for index in range(self.num_frames)],
                initial_value="All",
            )
        else:
            with self.server.gui.add_folder("Playback"):
                self.timestep = self.server.gui.add_slider(
                    "Timestep", min=0, max=self.num_frames - 1, step=1, initial_value=0
                )
                previous_button = self.server.gui.add_button("Previous Frame")
                next_button = self.server.gui.add_button("Next Frame")
                self.playing = self.server.gui.add_checkbox("Playing", initial_value=True)
                self.fps = self.server.gui.add_slider(
                    "FPS", min=1.0, max=60.0, step=0.5, initial_value=20.0
                )

            @previous_button.on_click
            def _(_) -> None:
                self.timestep.value = (int(self.timestep.value) - 1) % self.num_frames

            @next_button.on_click
            def _(_) -> None:
                self.timestep.value = (int(self.timestep.value) + 1) % self.num_frames

            @self.timestep.on_update
            def _(_) -> None:
                self.update_point_clouds()

        @self.conf_slider.on_update
        def _(_) -> None:
            self.update_point_clouds()

        if self.frame_selector is not None:
            @self.frame_selector.on_update
            def _(_) -> None:
                self.update_point_clouds()

    def _build_point_clouds(self) -> None:
        placeholder_points = np.zeros((1, 3), dtype=np.float32)
        placeholder_colors = np.zeros((1, 3), dtype=np.uint8)
        self.main_cloud = None
        self.static_cloud = None
        self.frame_cloud = None

        if self.mode == "static":
            self.main_cloud = self.server.scene.add_point_cloud(
                "points",
                points=placeholder_points,
                colors=placeholder_colors,
                point_size=0.001,
                point_shape="circle",
            )
        else:
            if self.mode == "dynamic":
                self.static_cloud = self.server.scene.add_point_cloud(
                    "static_points",
                    points=placeholder_points,
                    colors=placeholder_colors,
                    point_size=0.001,
                    point_shape="circle",
                )
            self.frame_cloud = self.server.scene.add_point_cloud(
                "dynamic_points" if self.mode == "dynamic" else "frame_points",
                points=placeholder_points,
                colors=placeholder_colors,
                point_size=0.001,
                point_shape="circle",
            )

    def confidence_mask(self) -> np.ndarray:
        confidence = self.confidence_by_frame
        finite = np.isfinite(confidence)
        if not finite.any():
            return np.zeros_like(confidence, dtype=bool)
        threshold = np.percentile(confidence[finite], self.conf_slider.value)
        return finite & (confidence >= threshold) & (confidence > 1e-5)

    @staticmethod
    def _set_cloud(handle, points: np.ndarray, colors: np.ndarray) -> None:
        finite = np.isfinite(points).all(axis=-1)
        points = points[finite]
        colors = colors[finite]
        if points.shape[0] == 0:
            points = np.zeros((1, 3), dtype=np.float32)
            colors = np.zeros((1, 3), dtype=np.uint8)
        handle.points = points
        handle.colors = colors

    def update_point_clouds(self) -> None:
        confidence_mask = self.confidence_mask()
        if self.mode == "static":
            if self.frame_selector.value == "All":
                keep = confidence_mask
                points = self.points_by_frame[keep]
                colors = self.colors_by_frame[keep]
            else:
                frame_idx = int(self.frame_selector.value)
                keep = confidence_mask[frame_idx]
                points = self.points_by_frame[frame_idx][keep]
                colors = self.colors_by_frame[frame_idx][keep]
            self._set_cloud(self.main_cloud, points, colors)
            return

        frame_idx = int(self.timestep.value)
        frame_keep = confidence_mask[frame_idx]
        if self.mode == "dynamic":
            static_keep = confidence_mask & self.static_masks_by_frame
            self._set_cloud(
                self.static_cloud,
                self.points_by_frame[static_keep],
                self.colors_by_frame[static_keep],
            )
            frame_keep &= ~self.static_masks_by_frame[frame_idx]
        self._set_cloud(
            self.frame_cloud,
            self.points_by_frame[frame_idx][frame_keep],
            self.colors_by_frame[frame_idx][frame_keep],
        )

    def _build_cameras(self) -> None:
        camera_to_world = closed_form_inverse_se3(self.extrinsics)[:, :3, :]
        camera_to_world[..., 3] -= self.scene_center
        self.camera_handles = []

        for frame_idx in range(self.num_frames):
            transform = viser_tf.SE3.from_matrix(camera_to_world[frame_idx])
            frame = self.server.scene.add_frame(
                f"camera_{frame_idx}",
                wxyz=transform.rotation().wxyz,
                position=transform.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            image = (self.images[frame_idx].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            focal_y = float(self.intrinsics[frame_idx, 1, 1])
            fov = 2.0 * np.arctan2(image.shape[0] / 2.0, focal_y)
            frustum = self.server.scene.add_camera_frustum(
                f"camera_{frame_idx}/frustum",
                fov=fov,
                aspect=image.shape[1] / image.shape[0],
                scale=0.05,
                image=image,
                line_width=1.0,
            )
            self.camera_handles.extend((frame, frustum))

            @frustum.on_click
            def _(_, frame=frame) -> None:
                for client in self.server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        @self.show_cameras.on_update
        def _(_) -> None:
            for handle in self.camera_handles:
                handle.visible = self.show_cameras.value

    def run(self) -> None:
        print(f"Viser is running at http://localhost:{self.server.get_port()}")
        try:
            while True:
                if self.playing is not None and self.playing.value:
                    self.timestep.value = (int(self.timestep.value) + 1) % self.num_frames
                    time.sleep(1.0 / self.fps.value)
                else:
                    time.sleep(0.05)
        except KeyboardInterrupt:
            print("Stopping Viser server")


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    image_paths = collect_images(args.input, args.sort_input)
    if args.mode == "dynamic" and len(image_paths) < 2:
        raise ValueError("--mode dynamic requires at least two images")

    need_point = args.use_point_map
    need_depth = not args.use_point_map or args.mode == "dynamic"
    print(f"Loading {len(image_paths)} images on {device}")
    model = load_model(
        args.checkpoint,
        device,
        teacher=args.teacher,
        need_depth=need_depth,
        need_point=need_point,
    )
    predictions = run_inference(
        model,
        image_paths,
        device,
        need_depth=need_depth,
        need_point=need_point,
    )

    if args.mode == "dynamic":
        print("Computing multi-view static masks")
        prepare_dynamic_mask(
            predictions,
            foreground_only=args.foreground_only,
            visibility_tolerance=args.visibility_tolerance,
        )

    if args.save_predictions is not None:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cpu_predictions(predictions), args.save_predictions)
        print(f"Saved predictions to {args.save_predictions}")

    predictions = numpy_predictions(predictions)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    demo = ViserDemo(
        predictions,
        mode=args.mode,
        use_point_map=args.use_point_map,
        port=args.port,
        conf_threshold=args.conf_threshold,
    )
    demo.run()


if __name__ == "__main__":
    main()
