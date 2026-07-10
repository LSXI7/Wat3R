from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def camera_centers(extrinsics):
    rotation = extrinsics[:, :, :3]
    translation = extrinsics[:, :, 3]
    return -np.einsum("nij,nj->ni", rotation.transpose(0, 2, 1), translation)


def umeyama_align(source, target):
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (target_centered.T @ source_centered) / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    scale = singular_values.sum() / ((source_centered**2).sum() / len(source))
    translation = target_mean - scale * (rotation @ source_mean)
    return (scale * (rotation @ source.T)).T + translation


def visualize_poses(gt_extrinsics, pred_extrinsics, sequence_name, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gt = camera_centers(gt_extrinsics)
    pred = umeyama_align(camera_centers(pred_extrinsics), gt)

    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(gt[:, 0], gt[:, 1], gt[:, 2], marker="o", label="GT")
    axis.plot(pred[:, 0], pred[:, 1], pred[:, 2], marker="^", linestyle="--", label="Wat3R")
    axis.set_title(f"Camera trajectory: {sequence_name}")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.legend()
    figure.savefig(output_dir / f"{sequence_name}_trajectory.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    errors = np.linalg.norm(gt - pred, axis=1)
    figure = plt.figure()
    plt.plot(np.arange(len(errors)), errors)
    plt.xlabel("Frame index")
    plt.ylabel("Camera-center error after alignment")
    plt.grid(True)
    figure.savefig(output_dir / f"{sequence_name}_error.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
