from __future__ import annotations

import torch

from wat3r.utils.geometry import (
    project_world_points_to_cam,
    unproject_depth_map_to_point_map_bs,
)


@torch.no_grad()
def depth_foreground_mask(
    depth: torch.Tensor,
    num_iters: int = 20,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
) -> torch.Tensor:
    """Split each depth map with 1-D K-means and return the nearer cluster."""
    if depth.ndim != 4:
        raise ValueError(f"Expected depth with shape (B, S, H, W), got {tuple(depth.shape)}")

    masks = torch.zeros_like(depth, dtype=torch.bool)
    flat_depth = depth.reshape(-1, depth.shape[-2] * depth.shape[-1])
    flat_masks = masks.reshape_as(flat_depth)

    for frame_idx, values in enumerate(flat_depth):
        valid = torch.isfinite(values) & (values > 0)
        samples = values[valid]
        if samples.numel() == 0:
            continue

        low, high = clip_percentiles
        low_value = torch.quantile(samples, low / 100.0)
        high_value = torch.quantile(samples, high / 100.0)
        samples = samples.clamp(low_value, high_value)

        center_a = samples.min()
        center_b = samples.max()
        for _ in range(num_iters):
            assign_a = (samples - center_a).abs() <= (samples - center_b).abs()
            if assign_a.any():
                center_a = samples[assign_a].mean()
            if (~assign_a).any():
                center_b = samples[~assign_a].mean()

        foreground_center = torch.minimum(center_a, center_b)
        background_center = torch.maximum(center_a, center_b)
        threshold = 0.5 * (foreground_center + background_center) * 0.9
        flat_masks[frame_idx] = valid & (values < threshold)

    return masks


@torch.no_grad()
def build_static_masks(
    depth: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    candidate_masks: torch.Tensor | None = None,
    visibility_tolerance: int = 2,
    relative_depth_threshold: float = 0.05,
    boundary: int = 4,
) -> torch.Tensor:
    """Find pixels whose reconstructed points are depth-consistent across views.

    A point is static when it is visible in at least ``S - visibility_tolerance``
    frames. For multi-frame inputs, at least two consistent views are required.
    """
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    if depth.ndim != 4:
        raise ValueError(f"Expected depth with shape (B, S, H, W), got {tuple(depth.shape)}")
    if visibility_tolerance < 0:
        raise ValueError("visibility_tolerance must be non-negative")

    batch_size, num_frames, height, width = depth.shape
    if num_frames < 2:
        raise ValueError("Dynamic separation requires at least two input frames")

    world_points, _, valid_depth = unproject_depth_map_to_point_map_bs(
        depth, extrinsics, intrinsics
    )
    if candidate_masks is None:
        candidate_masks = valid_depth
    else:
        if candidate_masks.shape != depth.shape:
            raise ValueError(
                f"Candidate masks must have shape {tuple(depth.shape)}, "
                f"got {tuple(candidate_masks.shape)}"
            )
        candidate_masks = candidate_masks.bool() & valid_depth

    min_visible_views = max(2, num_frames - visibility_tolerance)
    min_visible_views = min(min_visible_views, num_frames)
    output = torch.zeros(
        (batch_size, num_frames, height, width),
        device=depth.device,
        dtype=torch.bool,
    )

    for batch_idx in range(batch_size):
        for query_idx in range(num_frames):
            output[batch_idx, query_idx] = _build_frame_static_mask(
                extrinsics=extrinsics[batch_idx],
                intrinsics=intrinsics[batch_idx],
                world_points=world_points[batch_idx],
                depths=depth[batch_idx],
                candidate_mask=candidate_masks[batch_idx, query_idx],
                query_idx=query_idx,
                min_visible_views=min_visible_views,
                relative_depth_threshold=relative_depth_threshold,
                boundary=boundary,
            )

    return output


def _build_frame_static_mask(
    *,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    world_points: torch.Tensor,
    depths: torch.Tensor,
    candidate_mask: torch.Tensor,
    query_idx: int,
    min_visible_views: int,
    relative_depth_threshold: float,
    boundary: int,
) -> torch.Tensor:
    num_frames, height, width, _ = world_points.shape
    output = torch.zeros((height, width), device=world_points.device, dtype=torch.bool)
    if not candidate_mask.any():
        return output

    query_points = world_points[query_idx][candidate_mask]
    ys, xs = torch.nonzero(candidate_mask, as_tuple=True)
    image_points, camera_points = project_world_points_to_cam(
        query_points, extrinsics, intrinsics
    )
    projected_depth = camera_points[:, 2]
    pixel_floor = image_points.floor().long()

    inside = (
        (pixel_floor[..., 0] >= boundary)
        & (pixel_floor[..., 0] < width - boundary)
        & (pixel_floor[..., 1] >= boundary)
        & (pixel_floor[..., 1] < height - boundary)
        & torch.isfinite(projected_depth)
        & (projected_depth > 0)
    )
    safe_pixels = pixel_floor.clone()
    safe_pixels[~inside] = 0
    frame_indices = torch.arange(num_frames, device=depths.device)[:, None].expand(
        -1, safe_pixels.shape[1]
    )

    depth_consistent = torch.zeros_like(inside)
    for offset_x, offset_y in ((0, 0), (1, 0), (0, 1), (1, 1)):
        sample_pixels = safe_pixels + torch.tensor(
            (offset_x, offset_y), device=safe_pixels.device
        )
        sampled_depth = depths[
            frame_indices, sample_pixels[..., 1], sample_pixels[..., 0]
        ]
        depth_difference = (projected_depth - sampled_depth).abs()
        consistent = (
            torch.isfinite(sampled_depth)
            & (sampled_depth > 0)
            & (depth_difference < projected_depth * relative_depth_threshold)
            & (depth_difference < sampled_depth * relative_depth_threshold)
        )
        depth_consistent |= consistent

    visible_count = (inside & depth_consistent).sum(dim=0)
    static_points = visible_count >= min_visible_views
    output[ys[static_points], xs[static_points]] = True
    return output
