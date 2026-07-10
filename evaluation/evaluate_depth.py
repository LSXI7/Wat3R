#!/usr/bin/env python3
"""Unified monocular and multiview depth evaluation for Wat3R."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.depth_eval_tools import metric
from evaluation.depth_eval_tools.depth_dataloader import get_data_loader
from evaluation.depth_eval_tools.get_models import get_depth_pair, get_depth_single, init_model
from evaluation.depth_eval_tools.metric import MetricTracker
from evaluation.depth_eval_tools.test_single import eval_multi, eval_single, eval_single_aligned


EVAL_METRICS = [
    "abs_relative_difference",
    "squared_relative_difference",
    "rmse_linear",
    "rmse_log",
    "log10",
    "delta1_acc",
    "delta2_acc",
    "delta3_acc",
    "i_rmse",
    "silog_rmse",
]
MONO_DATASETS = {"flsea_vi", "seathru", "flsea_stereo", "squid"}
MULTIVIEW_DATASETS = {"seathru_full", "flsea_stereo_full"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Wat3R depth predictions")
    parser.add_argument("--mode", choices=("mono", "multiview"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True, help="Wat3R checkpoint path")
    parser.add_argument("--dataset-root", help="Override the dataset directory")
    parser.add_argument("--output-dir", default="evaluation/outputs")
    parser.add_argument("--alignment", choices=("least_square", "least_square_disparity"),
                        default="least_square")
    parser.add_argument("--alignment-max-res", type=int)
    parser.add_argument("--skip", type=int, default=0,
                        help="Multiview sampling: keep one frame every skip+1 frames")
    parser.add_argument("--teacher", action="store_true",
                        help="Evaluate ema_models from a training checkpoint")
    parser.add_argument("--save-figs", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    args = parser.parse_args()

    supported = MONO_DATASETS if args.mode == "mono" else MULTIVIEW_DATASETS
    if args.dataset not in supported:
        parser.error(f"{args.mode} mode supports: {', '.join(sorted(supported))}")
    if args.mode == "mono" and args.skip:
        parser.error("--skip is only valid in multiview mode")
    return args


def select_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def make_trackers(metric_funcs):
    tracker = MetricTracker(*[function.__name__ for function in metric_funcs])
    groups = defaultdict(lambda: MetricTracker(*[function.__name__ for function in metric_funcs]))
    return tracker, groups


def write_csv_row(handle, identifier, values):
    handle.write(identifier + "," + ",".join(values) + "\n")


def save_results(output_dir, loader, metric_funcs, tracker, group_trackers):
    group_path = output_dir / "per_group_metrics.csv"
    with group_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group"] + [function.__name__ for function in metric_funcs])
        for group in sorted(group_trackers):
            result = group_trackers[group].result()
            writer.writerow([group] + [result[function.__name__] for function in metric_funcs])

    result = tracker.result()
    summary_path = output_dir / "eval_metrics.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"dataset: {loader.dataset.disp_name}\n")
        handle.write(f"dataset_root: {loader.dataset.filename_ls_path}\n")
        handle.write(f"min_depth: {loader.dataset.min_depth}\n")
        handle.write(f"max_depth: {loader.dataset.max_depth}\n")
        for function in metric_funcs:
            handle.write(f"{function.__name__}: {result[function.__name__]}\n")
    print(f"Saved evaluation results to {output_dir}")


def evaluate_one_image(
    image_path,
    depth_raw_ts,
    valid_mask_ts,
    *,
    model,
    device,
    metric_funcs,
    tracker,
    group_trackers,
    group,
    args,
    visual_root,
    result_name,
    figure_index,
):
    if depth_raw_ts.max().item() == 0 or valid_mask_ts.sum().item() <= 10:
        return figure_index, None
    depth_raw = depth_raw_ts.cpu().numpy()
    height, width = depth_raw.shape
    prediction, disparity = get_depth_single(model, device, image_path)
    figure_index, values, _, _ = eval_single(
        depth_pred=prediction,
        disparity=disparity,
        H=height,
        W=width,
        depth_raw=depth_raw,
        i=figure_index,
        valid_mask_ts=valid_mask_ts,
        depth_raw_ts=depth_raw_ts,
        metric_tracker=tracker,
        group_trackers=group_trackers,
        save_figs=args.save_figs,
        output_dir=str(visual_root),
        after_result=result_name,
        device=device,
        alignment=args.alignment,
        alignment_max_res=args.alignment_max_res,
        min_depth=tracker.dataset_min_depth,
        max_depth=tracker.dataset_max_depth,
        metric_funcs=metric_funcs,
        gid=group,
    )
    return figure_index, values


def evaluate_mono(args, model, device, loader, output_dir, visual_root, result_name, metric_funcs):
    tracker, group_trackers = make_trackers(metric_funcs)
    tracker.dataset_min_depth = loader.dataset.min_depth
    tracker.dataset_max_depth = loader.dataset.max_depth
    csv_path = output_dir / "per_sample_metrics.csv"
    figure_index = 1
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("filename," + ",".join(EVAL_METRICS) + "\n")
        for data in tqdm(loader, desc=f"Monocular depth: {args.dataset}"):
            scene, obj = data["scene"][0], data["obj"][0]
            group = f"{scene}+{obj}"
            if args.dataset in {"flsea_stereo", "squid"}:
                views = (
                    (data["left_img"][0], data["depth_left"][0], data["valid_mask_left"][0]),
                    (data["right_img"][0], data["depth_right"][0], data["valid_mask_right"][0]),
                )
            else:
                views = ((data["image_path"][0], data["depth"][0], data["valid_mask"][0]),)

            for image_path, depth, mask in views:
                tracker.dataset_min_depth = loader.dataset.min_depth
                figure_index, values = evaluate_one_image(
                    image_path, depth, mask,
                    model=model, device=device, metric_funcs=metric_funcs,
                    tracker=tracker, group_trackers=group_trackers, group=group,
                    args=args, visual_root=visual_root, result_name=result_name,
                    figure_index=figure_index,
                )
                if values is not None:
                    write_csv_row(handle, f"{group}/{Path(image_path).name}", values)
    save_results(output_dir, loader, metric_funcs, tracker, group_trackers)


def evaluate_multiview(args, model, device, loader, output_dir, visual_root, result_name, metric_funcs):
    tracker, group_trackers = make_trackers(metric_funcs)
    sample_csv = (output_dir / "per_sample_metrics.csv").open("w", encoding="utf-8")
    image_csv = (output_dir / "per_image_metrics.csv").open("w", encoding="utf-8")
    header = "filename," + ",".join(EVAL_METRICS) + "\n"
    sample_csv.write(header)
    image_csv.write(header)
    figure_index = 1
    try:
        for sample_index, data in enumerate(tqdm(loader, desc=f"Multiview depth: {args.dataset}")):
            image_paths = [value[0] for value in data["image_path"]]
            if not image_paths:
                continue
            scene, obj = data["scene"][0][0], data["obj"][0][0]
            group = f"{scene}+{obj}"
            depth_raw_ts = torch.stack([value[0] for value in data["depth_ts"]])
            valid_mask_ts = torch.stack([value[0] for value in data["valid_mask"]])
            depth_raw = np.stack([value[0].cpu().numpy() for value in data["depth_np"]])
            prediction = get_depth_pair(model, device, image_paths, return_full=True)
            _, height, width = valid_mask_ts.shape

            figure_index, values, _, _, aligned_prediction = eval_multi(
                depth_pred_ori=prediction,
                H=height,
                W=width,
                depth_raw=depth_raw,
                i=figure_index,
                valid_mask_ts=valid_mask_ts,
                depth_raw_ts=depth_raw_ts,
                metric_tracker=tracker,
                group_trackers=group_trackers,
                save_figs=False,
                output_dir=str(visual_root),
                after_result=result_name,
                device=device,
                alignment=args.alignment,
                alignment_max_res=args.alignment_max_res,
                min_depth=loader.dataset.min_depth,
                max_depth=loader.dataset.max_depth,
                metric_funcs=metric_funcs,
                gid=group,
            )
            write_csv_row(sample_csv, f"{group}/{sample_index}", values)

            for index, image_path in enumerate(image_paths):
                figure_index, image_values = eval_single_aligned(
                    depth_pred=aligned_prediction[index],
                    depth_raw=depth_raw[index],
                    i=figure_index,
                    valid_mask_ts=valid_mask_ts[index],
                    depth_raw_ts=depth_raw_ts[index],
                    save_figs=args.save_figs,
                    output_dir=str(visual_root),
                    after_result=result_name,
                    device=device,
                    min_depth=loader.dataset.min_depth,
                    max_depth=loader.dataset.max_depth,
                    metric_funcs=metric_funcs,
                    name=Path(image_path).stem,
                )
                write_csv_row(image_csv, f"{group}/{Path(image_path).name}", image_values)
    finally:
        sample_csv.close()
        image_csv.close()
    save_results(output_dir, loader, metric_funcs, tracker, group_trackers)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    device = select_device(args.device)
    print(f"Using device: {device}")

    model = init_model(args.checkpoint, device, teacher=args.teacher)
    loader = get_data_loader(args.dataset, dataset_path=args.dataset_root, skip=args.skip)
    metric_funcs = [getattr(metric, name) for name in EVAL_METRICS]

    task_name = "mono_depth" if args.mode == "mono" else "multiview_depth"
    dataset_name = args.dataset + (f"_skip_{args.skip}" if args.mode == "multiview" and args.skip else "")
    result_name = Path(args.checkpoint).stem + ("_teacher" if args.teacher else "_student")
    task_root = Path(args.output_dir) / task_name / dataset_name
    output_dir = task_root / result_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "mono":
        evaluate_mono(args, model, device, loader, output_dir, task_root, result_name, metric_funcs)
    else:
        evaluate_multiview(args, model, device, loader, output_dir, task_root, result_name, metric_funcs)


if __name__ == "__main__":
    main()
