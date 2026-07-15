from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np


def _low_freq_field(shape, strength: float = 0.3, seed: Optional[int] = None) -> np.ndarray:
    """Generate a smooth multiplicative field in [1 - strength, 1 + strength]."""
    rng = np.random.default_rng(seed) if seed is not None else np.random
    height, width, channels = shape
    field = rng.normal(0, 1, size=(height, width, channels)).astype(np.float32)

    kernel = max(5, int(0.05 * min(height, width)) // 2 * 2 + 1)
    for _ in range(3):
        field = cv2.GaussianBlur(field, (kernel, kernel), kernel * 0.3, borderType=cv2.BORDER_REFLECT_101)

    field = field / (1e-6 + np.max(np.abs(field)))
    field = 1.0 + strength * field
    return field.astype(np.float32)


def sample_betas_physical(
    beta_scale: float = 1.0,
    beta_ratio_range=(0.6, 0.9),
    jitter: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample direct and backscatter attenuation coefficients in RGB order."""
    base = np.random.rand() * beta_scale
    beta_d = np.array([base * 1.5, base, base * 0.7], dtype=np.float32)
    beta_d *= 1.0 + np.random.normal(0, jitter, size=3).astype(np.float32)
    beta_b = beta_d * np.random.uniform(*beta_ratio_range)
    return beta_d.astype(np.float32), beta_b.astype(np.float32)


def _sample_random_background_color(jitter: float = 0.08) -> np.ndarray:
    """Sample a random background light color in RGB order."""
    color = np.random.rand(3).astype(np.float32)
    color += np.random.normal(0, jitter, size=3).astype(np.float32)
    return np.clip(color, 0.0, 1.0)


def _depth_from_relative(
    depth_rel: np.ndarray,
    z_min: float,
    z_max: float,
    z_median_target: Optional[float] = None,
) -> np.ndarray:
    """Map relative depth to a bounded effective water depth."""
    depth = depth_rel.astype(np.float32)

    finite = np.isfinite(depth)
    if np.any(finite):
        lo, hi = np.percentile(depth[finite], [2, 98])
    else:
        lo, hi = np.min(depth), np.max(depth)

    if hi <= lo:
        lo, hi = np.min(depth), np.max(depth)

    depth = np.clip(depth, lo, hi)
    if hi > lo:
        depth_norm = (depth - lo) / (hi - lo)
    else:
        depth_norm = np.zeros_like(depth, dtype=np.float32)

    depth_eff = z_min + depth_norm * (z_max - z_min)

    if z_median_target is not None:
        median = np.nanmedian(depth_eff)
        if np.isfinite(median):
            depth_eff = depth_eff + (z_median_target - median)

    return depth_eff.astype(np.float32)


def sample_water_parameters(
    beta_scale: float = 1.0,
    beta_ratio_range=(0.6, 0.9),
    beta_jitter: float = 0.1,
    background_jitter: float = 0.08,
    min_depth_range=(0.3, 2.0),
    max_depth_range=(10.0, 40.0),
) -> Dict[str, np.ndarray | float]:
    """Sample sequence-level parameters shared by all frames in one training sample."""
    beta_d_3, beta_b_3 = sample_betas_physical(
        beta_scale=beta_scale,
        beta_ratio_range=beta_ratio_range,
        jitter=beta_jitter,
    )
    return {
        "beta_D_3": beta_d_3,
        "beta_B_3": beta_b_3,
        "B_inf_3": _sample_random_background_color(jitter=background_jitter),
        "min_depth": float(np.random.uniform(*min_depth_range)),
        "max_depth": float(np.random.uniform(*max_depth_range)),
    }


def load_relative_depth_map(path: str, image_shape: tuple[int, int]) -> np.ndarray:
    """Load and resize a relative depth map used for water synthesis."""
    height, width = image_shape
    depth = np.load(path).squeeze(0).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    return cv2.resize(depth, (width, height))


def synthesize_underwater_image(
    image: np.ndarray,
    relative_depth: np.ndarray,
    water_params: Dict[str, np.ndarray | float],
    spatial_strength: Optional[float] = None,
    invalid_mask: Optional[np.ndarray] = None,
    invalid_depth: float = 100.0,
) -> np.ndarray:
    """Apply the current training-time underwater image formation model."""
    height, width = image.shape[:2]
    if spatial_strength is None:
        spatial_strength = float(np.random.uniform(0.05, 0.3))

    depth_for_water = _depth_from_relative(
        relative_depth,
        z_min=float(water_params["min_depth"]),
        z_max=float(water_params["max_depth"]),
    )
    if invalid_mask is not None:
        depth_for_water = depth_for_water.copy()
        depth_for_water[invalid_mask] = invalid_depth

    rng = np.random.default_rng()
    field = _low_freq_field((height, width, 3), strength=spatial_strength, seed=rng.integers(0, 1 << 31))
    field_c = _low_freq_field((height, width, 3), strength=spatial_strength * 0.5, seed=rng.integers(0, 1 << 31))

    beta_d = np.clip(np.asarray(water_params["beta_D_3"], dtype=np.float32).reshape(1, 1, 3) * field, 1e-4, None)
    beta_b = np.clip(np.asarray(water_params["beta_B_3"], dtype=np.float32).reshape(1, 1, 3) * field, 1e-4, None)
    background = np.clip(
        np.asarray(water_params["B_inf_3"], dtype=np.float32).reshape(1, 1, 3) * field_c,
        0.0,
        1.0,
    )

    clean = image.astype(np.float32) / 255.0
    z = depth_for_water[..., None].astype(np.float32)
    direct = np.exp(-beta_d * z)
    backscatter = np.exp(-beta_b * z)
    water = clean * direct + background * (1.0 - backscatter)
    return (np.clip(water, 0.0, 1.0) * 255.0).astype(np.uint8)
