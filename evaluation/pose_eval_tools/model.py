from __future__ import annotations

import torch

from wat3r.utils.load_fn import load_and_preprocess_images
from wat3r.utils.pose_enc import pose_encoding_to_extri_intri


def load_model(checkpoint_path, device, teacher=False):
    from wat3r.models.wat3r import Wat3R

    model = Wat3R(enable_track=False, enable_point=False, enable_depth=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if teacher:
        if "ema_models" not in checkpoint:
            raise KeyError("--teacher requires ema_models in the checkpoint")
        state_dict = checkpoint["ema_models"]
    else:
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    return model.to(device).eval()


@torch.no_grad()
def predict_extrinsics(model, image_paths, device):
    dtype = torch.float32
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    images = load_and_preprocess_images(image_paths, mode="max", target_size=518).to(device)
    with torch.cuda.amp.autocast(dtype=dtype, enabled=device.type == "cuda"):
        predictions = model(images, need_depth=False, need_point=False)
    extrinsics, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    return extrinsics[0].detach().cpu().numpy()
