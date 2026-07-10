#!/usr/bin/env python3
"""Evaluate multiview camera pose estimation on SeaThru-NeRF."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.pose_eval_tools.dataset import load_seathru_nerf
from evaluation.pose_eval_tools.model import load_model, predict_extrinsics
from evaluation.pose_eval_tools.visualization import visualize_poses
from wat3r.utils.geometry import closed_form_inverse_se3
from wat3r.utils.rotation import mat_to_quat


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Wat3R camera poses")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=("seathru_nerf",), default="seathru_nerf")
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir", default="evaluation/outputs")
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--save-figs", action="store_true")
    parser.add_argument("--random", action="store_true", help="Deterministically shuffle each sequence")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def select_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_pair_index(number):
    return torch.combinations(torch.arange(number), 2, with_replacement=False).unbind(-1)


def rotation_angle(gt_rotation, pred_rotation, eps=1e-15):
    pred_quaternion = mat_to_quat(pred_rotation)
    gt_quaternion = mat_to_quat(gt_rotation)
    loss = (1 - (pred_quaternion * gt_quaternion).sum(dim=1) ** 2).clamp(min=eps)
    return torch.arccos(1 - 2 * loss) * 180 / np.pi


def translation_angle(gt_translation, pred_translation, eps=1e-15):
    pred_translation = pred_translation / (torch.norm(pred_translation, dim=1, keepdim=True) + eps)
    gt_translation = gt_translation / (torch.norm(gt_translation, dim=1, keepdim=True) + eps)
    loss = torch.clamp_min(1.0 - torch.sum(pred_translation * gt_translation, dim=1) ** 2, eps)
    error = torch.acos(torch.sqrt(1 - loss)) * 180 / np.pi
    error[torch.isnan(error) | torch.isinf(error)] = 1e6
    return torch.minimum(error, torch.abs(180 - error))


def relative_pose_errors(predicted, target):
    first, second = build_pair_index(len(predicted))
    target_relative = target[first].bmm(closed_form_inverse_se3(target[second]))
    predicted_relative = predicted[first].bmm(closed_form_inverse_se3(predicted[second]))
    rotation = rotation_angle(target_relative[:, :3, :3], predicted_relative[:, :3, :3])
    translation = translation_angle(target_relative[:, :3, 3], predicted_relative[:, :3, 3])
    return rotation.cpu().numpy(), translation.cpu().numpy()


def calculate_auc(rotation_error, translation_error, threshold):
    maximum = np.maximum(rotation_error, translation_error)
    histogram, _ = np.histogram(maximum, bins=np.arange(threshold + 1))
    return float(np.mean(np.cumsum(histogram.astype(float) / len(maximum))))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    print(f"Using device: {device}")

    sequences = load_seathru_nerf(args.dataset_root, args.random, args.seed)
    model = load_model(args.checkpoint, device, teacher=args.teacher)
    result_name = Path(args.checkpoint).stem + ("_teacher" if args.teacher else "_student")
    output_dir = Path(args.output_dir) / "pose" / args.dataset / result_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sequence_name, frames in sequences.items():
        image_paths = [str(frame[0]) for frame in frames]
        target = np.stack([frame[1][:3] for frame in frames])
        predicted = predict_extrinsics(model, image_paths, device)
        if args.save_figs:
            visualize_poses(target, predicted, sequence_name, output_dir / "visualizations")

        target_tensor = torch.from_numpy(target).to(device)
        predicted_tensor = torch.from_numpy(predicted).to(device)
        bottom = torch.tensor([0, 0, 0, 1], device=device, dtype=target_tensor.dtype).expand(len(frames), 1, 4)
        rotation_error, translation_error = relative_pose_errors(
            torch.cat((predicted_tensor, bottom), dim=1),
            torch.cat((target_tensor, bottom), dim=1),
        )
        row = {
            "sequence": sequence_name,
            "AUC@30": calculate_auc(rotation_error, translation_error, 30),
            "AUC@15": calculate_auc(rotation_error, translation_error, 15),
            "AUC@5": calculate_auc(rotation_error, translation_error, 5),
            "AUC@3": calculate_auc(rotation_error, translation_error, 3),
            "R_ACC@5": float(np.mean(rotation_error < 5)),
            "T_ACC@5": float(np.mean(translation_error < 5)),
        }
        rows.append(row)
        print(row)

    average = {"sequence": "Average"}
    for key in rows[0]:
        if key != "sequence":
            average[key] = float(np.mean([row[key] for row in rows]))
    rows.append(average)
    with (output_dir / "pose_eval.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved pose results to {output_dir}")


if __name__ == "__main__":
    main()
