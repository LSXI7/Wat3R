import torch
from wat3r.utils.load_fn import load_and_preprocess_images


def init_model(model_path, device, teacher=False):
    from wat3r.models.wat3r import Wat3R
    model = Wat3R(enable_track=False, enable_camera=False, enable_point=False)
    state_dict = torch.load(model_path, map_location=device)
    if teacher:
        assert 'ema_models' in state_dict.keys()
        print('loading teacher')
        state_dict = state_dict['ema_models']
    elif 'model' in state_dict.keys():
        state_dict = state_dict['model']

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    else:
        print('ALL keys loaded.')
    model.to(device).eval()

    return model




@torch.no_grad()
def get_depth_single(model, device, image_path):
    disp = None
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    images = load_and_preprocess_images([image_path], mode='max', target_size=518).to(device)
    with torch.cuda.amp.autocast(dtype=dtype, enabled=device.type == 'cuda'):
        predictions = model(images, need_camera=False, need_point=False)
    depth_pred = predictions["depth"].squeeze(0).squeeze(0).squeeze(-1).cpu().numpy()
    return depth_pred, disp


@torch.no_grad()
def get_depth_pair(model, device, image_paths, return_full=False):
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    images = load_and_preprocess_images(image_paths, mode='max', target_size=518).to(device)
    with torch.cuda.amp.autocast(dtype=dtype, enabled=device.type == 'cuda'):
        predictions = model(images, need_camera=False, need_point=False)
    depth = predictions["depth"].squeeze(0).squeeze(-1).cpu().numpy()
    return depth if return_full else [item for item in depth]
