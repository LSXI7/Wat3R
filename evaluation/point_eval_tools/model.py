from __future__ import annotations

import torch

from wat3r.models.wat3r import Wat3R
from wat3r.utils.geometry import unproject_depth_map_to_point_map
from wat3r.utils.pose_enc import pose_encoding_to_extri_intri


def load_wat3r_model(checkpoint: str, device: torch.device, *, test_mode: int = 2, teacher: bool = False):
    if test_mode == 0:
        model = Wat3R(enable_track=False, enable_camera=False, enable_depth=False, enable_point=True)
    elif test_mode == 2:
        model = Wat3R(enable_track=False, enable_camera=True, enable_depth=True, enable_point=False)
    else:
        raise ValueError("--test-mode must be 0 or 2")

    state_dict = torch.load(checkpoint, map_location="cpu")
    if teacher:
        if "ema_models" not in state_dict:
            raise KeyError("--teacher requires a checkpoint containing 'ema_models'")
        print("Loading teacher weights from ema_models")
        state_dict = state_dict["ema_models"]
    elif isinstance(state_dict, dict) and "model" in state_dict:
        print("Loading student weights from model")
        state_dict = state_dict["model"]

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    ignored_missing = ("camera_head.", "point_head.", "depth_head.")
    missing_keys = [key for key in missing_keys if not key.startswith(ignored_missing)]
    unexpected_keys = [key for key in unexpected_keys if not key.startswith("track_head.")]
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    return model.to(device).eval()


@torch.no_grad()
def predict_world_points(model, views, device: torch.device, *, test_mode: int = 2, frames_chunk_size: int = 8):
    images = torch.stack([view["img"] for view in views], dim=0).unsqueeze(0).to(device)
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    need_point = test_mode == 0
    need_depth = test_mode == 2
    need_camera = test_mode == 2
    with torch.cuda.amp.autocast(dtype=dtype, enabled=device.type == "cuda"):
        predictions = model(
            images,
            frames_chunk_size=frames_chunk_size,
            need_point=need_point,
            need_depth=need_depth,
            need_camera=need_camera,
        )

    if test_mode == 0:
        world_points = predictions["world_points"].squeeze(0)
        world_points_conf = predictions["world_points_conf"].squeeze(0)
    elif test_mode == 2:
        image_size = images.shape[-2:]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], image_size)
        depth_map = predictions["depth"].squeeze(0)
        depth_conf = predictions["depth_conf"].squeeze(0)
        world_points = unproject_depth_map_to_point_map(
            depth_map,
            extrinsics.squeeze(0),
            intrinsics.squeeze(0),
        )
        world_points = torch.from_numpy(world_points).to(device=device, dtype=torch.float32)
        world_points_conf = depth_conf
    else:
        raise ValueError("--test-mode must be 0 or 2")

    return world_points, world_points_conf

