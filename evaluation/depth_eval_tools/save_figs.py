from pathlib import Path
import numpy as np
import cv2
import matplotlib

def save_fig_(depth_pred, output_dir, after_result, i, depth_raw, name=None, valid_mask=None):
    """
    Save predicted and ground-truth depth visualizations.

    If valid_mask is provided, GT depth is normalized only on valid pixels and
    invalid pixels are rendered as white.
    """

    def _to_numpy(x):
        """Convert ndarray, tensor, or array-like input to numpy."""
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x

        try:
            import torch
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
        except ImportError:
            pass
        return np.asarray(x)

    def _prepare_mask(mask, shape_hw):
        """Convert a mask to a 2D boolean numpy array."""
        if mask is None:
            return None
        mask = _to_numpy(mask)

        # Remove singleton dimensions such as 1x1xHxW, 1xHxW, or HxWx1.
        while mask.ndim > 2:
            mask = np.squeeze(mask, axis=0)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]

        if mask.shape != shape_hw:
            raise ValueError(f"valid_mask shape {mask.shape} does not match depth shape {shape_hw}")

        mask = mask.astype(bool)
        return mask

    def _depth_to_bgr(depth, cmap, mask=None):
        """Convert depth to a BGR colormap image."""
        depth = _to_numpy(depth)
        depth = np.squeeze(depth)
        # dmin = depth.min()
        dmax = depth.max()
        depth=dmax-depth
        if depth.ndim != 2:
            raise ValueError(f"depth must be 2D (H, W), but got shape {depth.shape}")

        h, w = depth.shape
        mask = _prepare_mask(mask, (h, w)) if mask is not None else None

        if mask is not None:
            # Empty masks are rendered as white images.
            if not np.any(mask):
                return np.full((h, w, 3), 255, dtype=np.uint8)

            depth_valid = depth[mask]
            dmin = depth_valid.min()
            dmax = depth_valid.max()

            depth_norm = np.zeros_like(depth, dtype=np.float32)
            if dmax - dmin > 1e-6:
                depth_norm[mask] = (depth[mask] - dmin) / (dmax - dmin)
        else:
            dmin = depth.min()
            dmax = depth.max()
            if dmax - dmin > 1e-6:
                depth_norm = (depth - dmin) / (dmax - dmin)
            else:
                depth_norm = np.zeros_like(depth, dtype=np.float32)

        depth_uint8 = (depth_norm * 255).astype(np.uint8)

        # matplotlib returns RGB, while cv2 writes BGR.
        colored_depth = (cmap(depth_uint8)[:, :, :3] * 255).astype(np.uint8)

        if mask is not None:
            colored_depth[~mask] = 255

        colored_depth_bgr = cv2.cvtColor(colored_depth, cv2.COLOR_RGB2BGR)
        return colored_depth_bgr

    def _build_path(prefix):
        """Build the output image path."""
        if name is None:
            filename = f"{prefix}_{i}.jpg"
        else:
            filename = f"{prefix}_{name}.jpg"
        return Path(output_dir) / "visual" / after_result / filename

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    # Save predicted depth without masking.
    pred_img = _depth_to_bgr(depth_pred, cmap, mask=None)
    pred_path = _build_path("depth")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    print("saving pred depth to", pred_path)
    cv2.imwrite(str(pred_path), pred_img)

    # Save GT depth with invalid pixels rendered as white.
    gt_img = _depth_to_bgr(depth_raw, cmap, mask=valid_mask)
    gt_path = _build_path("depth_raw")
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    print("saving raw depth to", gt_path)
    cv2.imwrite(str(gt_path), gt_img)
