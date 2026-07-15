from __future__ import annotations

import torch
import torch.nn.functional as F

from training.losses.ema_loss import regression_loss
from wat3r.utils.geometry import closed_form_inverse_se3


def _build_pixel_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=-1)
    return grid.unsqueeze(0).unsqueeze(0).float()


def _backproject_depth(depth_ref: torch.Tensor, intrinsics_ref: torch.Tensor, pixel_grid: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = depth_ref.shape
    grid = pixel_grid.to(device=depth_ref.device, dtype=depth_ref.dtype).expand(batch_size, -1, -1, -1, -1)
    intrinsics_ref = intrinsics_ref.to(device=depth_ref.device, dtype=depth_ref.dtype)
    u = grid[..., 0]
    v = grid[..., 1]

    fx = intrinsics_ref[:, 0, 0].view(batch_size, 1, 1, 1).clamp_min(1e-8)
    fy = intrinsics_ref[:, 1, 1].view(batch_size, 1, 1, 1).clamp_min(1e-8)
    cx = intrinsics_ref[:, 0, 2].view(batch_size, 1, 1, 1)
    cy = intrinsics_ref[:, 1, 2].view(batch_size, 1, 1, 1)

    z = depth_ref
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return torch.cat([x, y, z], dim=1)


def _project_teacher_depth_to_student_views(
    teacher_depth: torch.Tensor,
    student_depth: torch.Tensor,
    student_conf: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    static_mask: torch.Tensor,
    ref_idx: int,
    detach_teacher: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if student_depth.ndim == 5 and student_depth.shape[2] != 1:
        batch_size, num_views, height, width, _ = student_depth.shape
        student_depth = student_depth.permute(0, 1, 4, 2, 3).contiguous()
        teacher_depth = teacher_depth.permute(0, 1, 4, 2, 3).contiguous()
    else:
        batch_size, num_views, _, height, width = student_depth.shape
    if student_conf.ndim == 5:
        if student_conf.shape[-1] == 1:
            student_conf = student_conf[..., 0]
        elif student_conf.shape[2] == 1:
            student_conf = student_conf[:, :, 0]

    device = student_depth.device
    student_depth = student_depth.to(torch.float32)
    student_conf = student_conf.to(device=device, dtype=torch.float32)
    teacher_depth = teacher_depth.to(device=device, dtype=torch.float32)
    intrinsics = intrinsics.to(device=device, dtype=torch.float32)
    extrinsics = extrinsics.to(device=device, dtype=torch.float32)
    mask_bool = static_mask.to(device=device).bool()
    extrinsics_flat = extrinsics.reshape(batch_size * num_views, 3, 4)
    cam_to_world = closed_form_inverse_se3(extrinsics_flat)[:, :3].reshape(batch_size, num_views, 3, 4)
    rotations = cam_to_world[..., :3]
    translations = cam_to_world[..., 3:]
    pixel_grid = _build_pixel_grid(height, width, device)

    context = torch.no_grad() if detach_teacher else torch.enable_grad()
    with context:
        teacher_depth_ref = teacher_depth[:, ref_idx]
        intrinsics_ref = intrinsics[:, ref_idx]
        cam_to_world_ref = cam_to_world[:, ref_idx]
        mask_ref = mask_bool[:, ref_idx]

        ref_points = _backproject_depth(teacher_depth_ref, intrinsics_ref, pixel_grid)
        ref_points_flat = ref_points.reshape(batch_size, 3, -1)
        world_points = cam_to_world_ref[..., :3] @ ref_points_flat + cam_to_world_ref[..., 3:]

        world_points = world_points.unsqueeze(1)
        target_points = torch.matmul(rotations.transpose(-1, -2), world_points - translations)
        projected = torch.matmul(intrinsics, target_points)
        projected_depth = target_points[:, :, 2, :]

        safe_depth = projected_depth.clamp_min(1e-6)
        x = (projected[:, :, 0, :] / safe_depth).reshape(batch_size, num_views, height, width)
        y = (projected[:, :, 1, :] / safe_depth).reshape(batch_size, num_views, height, width)
        x_norm = (x / (width - 1)) * 2 - 1
        y_norm = (y / (height - 1)) * 2 - 1
        sample_grid = torch.stack([x_norm, y_norm], dim=-1).reshape(batch_size * num_views, height, width, 2)
        projected_depth = projected_depth.reshape(batch_size, num_views, height, width)

    student_depth_flat = student_depth[:, :, 0].reshape(batch_size * num_views, 1, height, width)
    sample_grid = sample_grid.float()
    sampled_student_depth = F.grid_sample(
        student_depth_flat,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0].reshape(batch_size, num_views, height, width)

    student_conf_flat = student_conf.reshape(batch_size * num_views, 1, height, width)
    sampled_student_conf = F.grid_sample(
        student_conf_flat,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0].reshape(batch_size, num_views, height, width)

    mask_flat = mask_bool.reshape(batch_size * num_views, 1, height, width).to(dtype=sample_grid.dtype)
    sampled_mask = F.grid_sample(
        mask_flat,
        sample_grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0].reshape(batch_size, num_views, height, width).bool()

    weight = mask_ref.unsqueeze(1) & sampled_mask & (projected_depth > 0)
    view_ids = torch.arange(num_views, device=device).view(1, num_views, 1, 1)
    weight = weight & (view_ids != ref_idx)
    return weight, projected_depth, sampled_student_depth, sampled_student_conf


def static_geometry_loss(
    teacher_out: dict,
    student_out: dict,
    static_mask: torch.Tensor,
    gamma: float = 1.0,
    alpha: float = 0.2,
    gradient_loss_fn: str | None = None,
    valid_range: float = -1,
    **kwargs,
) -> dict[str, torch.Tensor]:
    student_depth = student_out["depth"]
    teacher_depth = teacher_out["depth"]
    student_conf = student_out["depth_conf"]
    _, num_views, _, _, _ = student_depth.shape

    loss_conf_all = 0.0
    loss_grad_all = 0.0
    loss_reg_all = 0.0

    for ref_idx in range(num_views):
        weight, projected_depth, sampled_student_depth, sampled_student_conf = _project_teacher_depth_to_student_views(
            teacher_depth=teacher_depth,
            student_depth=student_depth,
            student_conf=student_conf,
            intrinsics=teacher_out["intrinsics"],
            extrinsics=teacher_out["extrinsics"],
            static_mask=static_mask,
            ref_idx=ref_idx,
        )
        if weight.sum() < 100:
            dummy = (0.0 * student_depth).mean()
            loss_conf = dummy
            loss_grad = dummy
            loss_reg = dummy
        else:
            loss_conf, loss_grad, loss_reg = regression_loss(
                sampled_student_depth.unsqueeze(-1),
                projected_depth.to(torch.float32).unsqueeze(-1),
                weight,
                conf=sampled_student_conf,
                gradient_loss_fn=gradient_loss_fn,
                gamma=gamma,
                alpha=alpha,
                valid_range=valid_range,
            )
        loss_conf_all = loss_conf_all + loss_conf
        loss_grad_all = loss_grad_all + loss_grad
        loss_reg_all = loss_reg_all + loss_reg

    loss_conf_all = loss_conf_all / num_views
    loss_grad_all = loss_grad_all / num_views
    loss_reg_all = loss_reg_all / num_views
    loss_total = loss_conf_all + loss_grad_all + loss_reg_all

    return {
        "ema_loss_static_geo": loss_total,
        "ema_loss_conf_static_geo": loss_conf_all,
        "ema_loss_grad_static_geo": loss_grad_all,
        "ema_loss_reg_static_geo": loss_reg_all,
    }
