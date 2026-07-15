# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging
from typing import Optional, Tuple
from wat3r.utils.geometry import closed_form_inverse_se3
from training.train_utils.general import check_and_fix_inf_nan


def check_valid_tensor(input_tensor: Optional[torch.Tensor], name: str = "tensor") -> None:
    """
    Check if a tensor contains NaN or Inf values and log a warning if found.
    
    Args:
        input_tensor: The tensor to check
        name: Name of the tensor for logging purposes
    """
    if input_tensor is not None:
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            logging.warning(f"NaN or Inf found in tensor: {name}")


def normalize_camera_extrinsics_and_points_batch(
    extrinsics: torch.Tensor,
    cam_points: Optional[torch.Tensor] = None,
    world_points: Optional[torch.Tensor] = None,
    depths: Optional[torch.Tensor] = None,
    scale_by_points: bool = True,
    point_masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Normalize camera extrinsics and corresponding 3D points.
    
    This function transforms the coordinate system to be centered at the first camera
    and optionally scales the scene to have unit average distance.
    
    Args:
        extrinsics: Camera extrinsic matrices of shape (B, S, 3, 4)
        cam_points: 3D points in camera coordinates of shape (B, S, H, W, 3) or (*,3)
        world_points: 3D points in world coordinates of shape (B, S, H, W, 3) or (*,3)
        depths: Depth maps of shape (B, S, H, W)
        scale_by_points: Whether to normalize the scale based on point distances
        point_masks: Boolean masks for valid points of shape (B, S, H, W)
    
    Returns:
        Tuple containing:
        - Normalized camera extrinsics of shape (B, S, 3, 4)
        - Normalized camera points (same shape as input cam_points)
        - Normalized world points (same shape as input world_points)
        - Normalized depths (same shape as input depths)
    """
    # Validate inputs
    check_valid_tensor(extrinsics, "extrinsics")
    check_valid_tensor(cam_points, "cam_points")
    check_valid_tensor(world_points, "world_points")
    check_valid_tensor(depths, "depths")


    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    # assert device == torch.device("cpu")


    # Convert extrinsics to homogeneous form: (B, N,4,4)    # world→camera
    extrinsics_homog = torch.cat(
        [
            extrinsics,
            torch.zeros((B, S, 1, 4), device=device),
        ],
        dim=-2,
    )
    extrinsics_homog[:, :, -1, -1] = 1.0

    # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
    # which can be also viewed as the cam_to_world extrinsic matrix
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0]) # cam_to_world
    # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
    new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)


    if world_points is not None:
        # since we are transforming the world points to the first camera's coordinate system
        # we directly use the cam_from_world extrinsic matrix of the first camera
        # instead of using the inverse of the first camera's extrinsic matrix
        R = extrinsics[:, 0, :3, :3]
        t = extrinsics[:, 0, :3, 3]
        new_world_points = (world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        new_world_points = None


    if scale_by_points:
        new_cam_points = cam_points.clone()
        new_depths = depths.clone()

        dist = new_world_points.norm(dim=-1)
        dist_sum = (dist * point_masks).sum(dim=[1,2,3])
        valid_count = point_masks.sum(dim=[1,2,3])
        avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)
        print('avg_scale', avg_scale)

        new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        if depths is not None:
            new_depths = new_depths / avg_scale.view(-1, 1, 1, 1)
        if cam_points is not None:
            new_cam_points = new_cam_points / avg_scale.view(-1, 1, 1, 1, 1)
    else:
        return new_extrinsics[:, :, :3], cam_points, new_world_points, depths

    new_extrinsics = new_extrinsics[:, :, :3] # 4x4 -> 3x4
    new_extrinsics = check_and_fix_inf_nan(new_extrinsics, "new_extrinsics", hard_max=None)
    new_cam_points = check_and_fix_inf_nan(new_cam_points, "new_cam_points", hard_max=None)
    new_world_points = check_and_fix_inf_nan(new_world_points, "new_world_points", hard_max=None)
    new_depths = check_and_fix_inf_nan(new_depths, "new_depths", hard_max=None)


    return new_extrinsics, new_cam_points, new_world_points, new_depths

#
# def normalize_camera_extrinsics_and_points_batch_student(
#         t_out=None,
#         ema_scale_by_student: bool = True,
#         s_out=None,
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
#     """
#     Normalize camera extrinsics and corresponding 3D points.
#
#     This function transforms the coordinate system to be centered at the first camera
#     and optionally scales the scene to have unit average distance.
#
#     Args:
#         extrinsics: Camera extrinsic matrices of shape (B, S, 3, 4)
#         cam_points: 3D points in camera coordinates of shape (B, S, H, W, 3) or (*,3)
#         world_points: 3D points in world coordinates of shape (B, S, H, W, 3) or (*,3)
#         depths: Depth maps of shape (B, S, H, W)
#         scale_by_points: Whether to normalize the scale based on point distances
#         point_masks: Boolean masks for valid points of shape (B, S, H, W)
#
#     Returns:
#         Tuple containing:
#         - Normalized camera extrinsics of shape (B, S, 3, 4)
#         - Normalized world points (same shape as input world_points)
#         - Normalized depths (same shape as input depths)
#     """
#     teacher_extrinsics = t_out['extrinsics'].clone()
#     # cam_points: Optional[torch.Tensor] = None,
#     teacher_world_points = t_out['world_points'].clone()
#     teacher_depths = t_out['depth'].clone()
#     point_masks = t_out['point_masks'].clone()
#     # point_masks=torch.ones_like(depths,device=depths)
#     # Validate inputs
#     check_valid_tensor(teacher_extrinsics, "extrinsics")
#     check_valid_tensor(teacher_world_points, "world_points")
#     check_valid_tensor(teacher_depths, "depths")
#
#     B, S, H, W, _ = teacher_depths.shape
#
#     assert ema_scale_by_student is True
#     # if ema_scale_by_student:
#     student_points = s_out['world_points'].clone()
#     # new_depths = depths.clone()
#
#     teacher_dist = teacher_world_points.norm(dim=-1)
#     student_dist = student_points.norm(dim=-1)
#     teacher_dist_sum = (teacher_dist * point_masks).sum(dim=[1, 2, 3])
#     student_dist_sum = (student_dist * point_masks).sum(dim=[1, 2, 3])
#     # valid_count = H * W * S
#     valid_count = point_masks.sum(dim=[1, 2, 3])
#     teacher_avg_scale = (teacher_dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)
#     student_avg_scale = (student_dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)
#     print('teacher_avg_scale', teacher_avg_scale)
#     print('student_avg_scale', student_avg_scale)
#     new_avg_scale = teacher_avg_scale / student_avg_scale
#
#     teacher_world_points = teacher_world_points / new_avg_scale.view(-1, 1, 1, 1, 1)
#     teacher_extrinsics[:, :, :3, 3] = teacher_extrinsics[:, :, :3, 3] / new_avg_scale.view(-1, 1, 1)
#     # if teacher_depths is not None:
#     teacher_depths = teacher_depths / new_avg_scale.view(-1, 1, 1, 1, 1)
#     # else:
#     #     return new_extrinsics[:, :, :3], cam_points, new_world_points, depths
#
#     teacher_extrinsics = teacher_extrinsics[:, :, :3]  # 4x4 -> 3x4
#     teacher_extrinsics = check_and_fix_inf_nan(teacher_extrinsics, "new_extrinsics", hard_max=None)
#     teacher_world_points = check_and_fix_inf_nan(teacher_world_points, "new_world_points", hard_max=None)
#     teacher_depths = check_and_fix_inf_nan(teacher_depths, "new_depths", hard_max=None)
#
#     return teacher_extrinsics, teacher_world_points, teacher_depths



