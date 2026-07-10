#!/usr/bin/env python3
"""Wat3R point-cloud reconstruction evaluation on Water3D."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.point_eval_tools.dataset import Water3DManyViewDataset
from evaluation.point_eval_tools.model import load_wat3r_model, predict_world_points
from evaluation.point_eval_tools.recon_metric import accuracy, completion


METRIC_COLUMNS = [
    "metrics_time",
    "acc",
    "acc_median",
    "comp",
    "comp_median",
    "nc1",
    "nc1_median",
    "nc2",
    "nc2_median",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Wat3R point-cloud reconstruction on Water3D")
    parser.add_argument("--checkpoint", required=True, help="Wat3R checkpoint path")
    parser.add_argument("--dataset-root", default="evaluation/datasets/water3D")
    parser.add_argument("--output-dir", default="evaluation/outputs/point_cloud")
    parser.add_argument("--run-name", help="Override the output run directory name")
    parser.add_argument("--test-mode", type=int, choices=(0, 2), default=2,
                        help="0: point head, 2: depth+camera unprojection")
    parser.add_argument("--num-views", type=int, default=20)
    parser.add_argument("--scene", action="append", help="Evaluate only this scene; can be repeated")
    parser.add_argument("--max-scenes", type=int, help="Limit the number of scenes for debugging")
    parser.add_argument("--metric-conf-percentile", type=float, default=0.0)
    parser.add_argument("--icp-conf-percentile", type=float, default=85.0)
    parser.add_argument("--min-depth-percentile", type=float, default=2.0)
    parser.add_argument("--max-depth-percentile", type=float, default=98.0)
    parser.add_argument("--max-depth", type=float, default=-1.0)
    parser.add_argument("--frames-chunk-size", type=int, default=8)
    parser.add_argument("--teacher", action="store_true", help="Evaluate ema_models from a training checkpoint")
    parser.add_argument("--save-o3d", action="store_true", help="Save GT and predicted point clouds as PLY")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def output_directory(args) -> Path:
    checkpoint_name = Path(args.checkpoint).stem
    run_name = args.run_name or f"{checkpoint_name}_test_mode_{args.test_mode}"
    if args.teacher:
        run_name = f"teacher_{run_name}"
    output_dir = Path(args.output_dir) / "water3D" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_point_clouds(output_dir: Path, scene_name: str, pred_raw, pred_aligned, gt_points, colors):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Saving point clouds requires open3d. Install the demo/evaluation dependencies.") from exc

    scene_dir = output_dir / "visual" / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    def _write(path: Path, points, point_colors=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if point_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(point_colors)
        o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False)

    _write(scene_dir / "pred_raw.ply", pred_raw, colors)
    _write(scene_dir / "pred_aligned.ply", pred_aligned, colors)
    _write(scene_dir / "gt.ply", gt_points)


def evaluate_scene(
    views,
    *,
    model,
    device,
    args,
    output_dir: Path,
):
    try:
        import open3d as o3d
        import roma
    except ImportError as exc:
        raise ImportError(
            "Point-cloud evaluation requires open3d and roma. Install the demo/evaluation dependencies."
        ) from exc

    scene_name = views[0]["label"]
    world_points, world_points_conf = predict_world_points(
        model,
        views,
        device,
        test_mode=args.test_mode,
        frames_chunk_size=args.frames_chunk_size,
    )

    pred_pts_list = []
    gt_pts_icp_list = []
    gt_pts_metric_list = []
    pred_colors_list = []
    gt_colors_list = []
    weights_list = []

    for index, view in enumerate(views):
        pred_points = world_points[index].detach()
        confidence = world_points_conf[index].detach()
        gt_points = torch.from_numpy(view["pts3d"]).to(device)
        valid_mask = torch.from_numpy(view["valid_mask"]).to(device)

        conf_flat = confidence.reshape(-1)
        metric_threshold = torch.quantile(conf_flat, args.metric_conf_percentile / 100.0)
        icp_threshold = torch.quantile(conf_flat, args.icp_conf_percentile / 100.0)
        metric_conf_mask = confidence >= metric_threshold

        pred_mask = valid_mask & metric_conf_mask
        gt_metric_mask = valid_mask
        if pred_mask.sum().item() == 0 or gt_metric_mask.sum().item() == 0:
            continue

        pred_masked = pred_points[pred_mask]
        gt_icp_masked = gt_points[pred_mask]
        gt_metric_masked = gt_points[gt_metric_mask]
        confidence_masked = confidence[pred_mask]
        weights = confidence_masked >= icp_threshold

        image = view["img"].to(device).permute(1, 2, 0)
        pred_colors = image[pred_mask]
        gt_colors = image[gt_metric_mask]

        pred_pts_list.append(pred_masked)
        gt_pts_icp_list.append(gt_icp_masked)
        gt_pts_metric_list.append(gt_metric_masked)
        pred_colors_list.append(pred_colors)
        gt_colors_list.append(gt_colors)
        weights_list.append(weights)

    if not pred_pts_list or not gt_pts_metric_list:
        raise RuntimeError(f"No valid points for scene {scene_name}")

    pred_points = torch.cat(pred_pts_list, dim=0).float().cpu()
    gt_points_icp = torch.cat(gt_pts_icp_list, dim=0).float().cpu()
    gt_points_metric = torch.cat(gt_pts_metric_list, dim=0).float().cpu()
    pred_colors = torch.cat(pred_colors_list, dim=0).float().cpu()
    gt_colors = torch.cat(gt_colors_list, dim=0).float().cpu()
    weights = torch.cat(weights_list, dim=0).float().cpu()
    if pred_points.shape[0] < 3 or gt_points_metric.shape[0] < 3:
        raise RuntimeError(f"Not enough valid points for scene {scene_name}")
    if weights.sum().item() < 3:
        weights = torch.ones_like(weights)

    start = time.time()
    rotation, translation, scale = roma.rigid_points_registration(
        pred_points,
        gt_points_icp,
        weights=weights,
        compute_scaling=True,
    )
    pred_aligned = scale * (pred_points @ rotation.T) + translation

    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_aligned.numpy())
    pred_pcd.colors = o3d.utility.Vector3dVector(pred_colors.numpy())
    pred_pcd.estimate_normals()

    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points_metric.numpy())
    gt_pcd.colors = o3d.utility.Vector3dVector(gt_colors.numpy())
    gt_pcd.estimate_normals()

    pred_normals = np.asarray(pred_pcd.normals)
    gt_normals = np.asarray(gt_pcd.normals)
    pred_points_np = np.asarray(pred_pcd.points)
    gt_points_np = np.asarray(gt_pcd.points)

    acc, acc_median, nc1, nc1_median = accuracy(gt_points_np, pred_points_np, gt_normals, pred_normals)
    comp, comp_median, nc2, nc2_median = completion(gt_points_np, pred_points_np, gt_normals, pred_normals)
    metrics_time = time.time() - start

    if args.save_o3d:
        save_point_clouds(
            output_dir,
            scene_name,
            pred_points.numpy(),
            pred_points_np,
            gt_points_metric.numpy(),
            pred_colors.numpy(),
        )

    return {
        "scene_name": scene_name,
        "num_views": len(views),
        "num_pred_points": int(pred_points_np.shape[0]),
        "num_gt_points": int(gt_points_np.shape[0]),
        "metrics_time": metrics_time,
        "acc": float(acc),
        "acc_median": float(acc_median),
        "comp": float(comp),
        "comp_median": float(comp_median),
        "nc1": float(nc1),
        "nc1_median": float(nc1_median),
        "nc2": float(nc2),
        "nc2_median": float(nc2_median),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    summary = {"num_scenes": len(rows)}
    for key in METRIC_COLUMNS:
        summary[f"{key}_mean"] = float(np.mean([row[key] for row in rows]))
    return summary


def main():
    args = parse_args()
    device = select_device(args.device)
    output_dir = output_directory(args)
    print(f"Using device: {device}")
    print(f"Saving results to: {output_dir}")

    dataset = Water3DManyViewDataset(
        args.dataset_root,
        num_views=args.num_views,
        scenes=args.scene,
        min_depth_percentile=args.min_depth_percentile,
        max_depth_percentile=args.max_depth_percentile,
        max_depth=args.max_depth,
    )
    if args.max_scenes is not None:
        dataset.scenes = dataset.scenes[:args.max_scenes]

    model = load_wat3r_model(args.checkpoint, device, test_mode=args.test_mode, teacher=args.teacher)

    rows = []
    errors = []
    for index in tqdm(range(len(dataset)), desc="Point cloud evaluation"):
        views = dataset[index]
        scene_name = views[0]["label"]
        try:
            row = evaluate_scene(views, model=model, device=device, args=args, output_dir=output_dir)
        except Exception as exc:
            print(f"[ERROR] {scene_name}: {exc}")
            errors.append({"scene_name": scene_name, "error": repr(exc)})
            continue

        rows.append(row)
        print(
            f"{scene_name}: acc={row['acc']:.4f}, comp={row['comp']:.4f}, "
            f"nc1={row['nc1']:.4f}, nc2={row['nc2']:.4f}"
        )

        write_csv(
            output_dir / "metrics_by_scene.csv",
            rows,
            ["scene_name", "num_views", "num_pred_points", "num_gt_points", *METRIC_COLUMNS],
        )

    if not rows:
        raise RuntimeError("No scenes were evaluated successfully")

    write_csv(
        output_dir / "metrics_overall.csv",
        [summarize(rows)],
        ["num_scenes", *[f"{key}_mean" for key in METRIC_COLUMNS]],
    )
    if errors:
        write_csv(output_dir / "errors.csv", errors, ["scene_name", "error"])
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
