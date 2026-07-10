import os.path

import cv2
import matplotlib
import numpy as np
import torch
from .save_figs import save_fig_
from .alignment import (
    align_depth_least_square,
    depth2disparity,
    disparity2depth,
)

matplotlib.use('Agg')


def eval_single(depth_pred, H, W, depth_raw, i, valid_mask_ts, depth_raw_ts, metric_tracker, group_trackers, save_figs,
                output_dir, after_result, device, alignment, alignment_max_res, min_depth, max_depth, metric_funcs,
                gid, disparity=None):
    depth_pred = cv2.resize(depth_pred, (W, H), interpolation=cv2.INTER_NEAREST)

    # if save_figs:
    #     save_fig_(depth_pred=depth_pred, output_dir=output_dir, after_result=after_result, i=i,
    #               depth_raw=depth_raw)
    #
    # i += 1

    valid_mask = valid_mask_ts.numpy()

    depth_raw_ts = depth_raw_ts.to(device)
    print('depth_raw_ts max', depth_raw_ts.max().item())
    print('depth_raw_ts min', depth_raw_ts.min().item())
    valid_mask_ts = valid_mask_ts.to(device)
    print(f'Using {alignment} with {alignment_max_res} resolution')
    # Align with GT using least square
    if disparity is not None:
        alignment='least_square_disparity'
    if "least_square" == alignment:
        depth_pred, scale, shift = align_depth_least_square(
            gt_arr=depth_raw,
            pred_arr=depth_pred,
            valid_mask_arr=valid_mask,
            return_scale_shift=True,
            max_resolution=alignment_max_res,
        )
    elif "least_square_disparity" == alignment:
        # convert GT depth -> GT disparity
        print('depth_raw',depth_raw[valid_mask].max())
        print('depth_raw',depth_raw[valid_mask].min())
        gt_disparity, gt_non_neg_mask = depth2disparity(
            depth=depth_raw, return_mask=True
        )
        print('gt_disparity',gt_disparity[valid_mask].min())
        print('gt_disparity',gt_disparity[valid_mask].max())
        # LS alignment in disparity space
        pred_non_neg_mask = depth_pred > 0
        print('pred_non_neg_mask',pred_non_neg_mask.sum())
        valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

        disparity_pred, scale, shift = align_depth_least_square(
            gt_arr=gt_disparity,
            pred_arr=depth_pred,
            valid_mask_arr=valid_nonnegative_mask,
            return_scale_shift=True,
            max_resolution=alignment_max_res,
        )
        print('scale',scale)
        print('shift',shift)
        # convert to depth
        print('disparity_pred',disparity_pred.min())
        print('disparity_pred',disparity_pred.max())
        # disparity_pred=disparity_pred.max()-disparity_pred
        disparity_pred = np.clip(
            disparity_pred, a_min=1e-3, a_max=None
        )  # avoid 0 disparity
        depth_pred = disparity2depth(disparity_pred)
    else:
        raise NotImplementedError

    # Clip to dataset min max
    depth_pred = np.clip(
        depth_pred, a_min=min_depth, a_max=max_depth
    )

    if save_figs:
        save_fig_(depth_pred=depth_pred, output_dir=output_dir, after_result=after_result, i=i,
                  depth_raw=depth_raw,valid_mask=valid_mask)

    i += 1

    # Evaluate
    sample_metric = []
    depth_pred_ts = torch.from_numpy(depth_pred).to(device)

    for met_func in metric_funcs:
        _metric_name = met_func.__name__
        print('_metric_name', _metric_name)
        _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
        print('_metric', _metric)
        sample_metric.append(_metric.__str__())
        if metric_tracker is not None:
            metric_tracker.update(_metric_name, _metric)
        if group_trackers is not None:
            group_trackers[gid].update(_metric_name, _metric)

    return i, sample_metric, metric_tracker, group_trackers


def eval_single_v2(depth_pred, H, W, depth_raw, i, valid_mask_ts, depth_raw_ts, metric_tracker, group_trackers, save_figs,
                output_dir, after_result, device, alignment, alignment_max_res, min_depth, max_depth, metric_funcs,
                gid, disparity=None,name=None):

    if save_figs:
        name_=os.path.join(name+'no_resize')
        save_fig_(depth_pred=depth_pred, output_dir=output_dir, after_result=after_result, i=i,
                  depth_raw=depth_raw,name=name_,valid_mask=valid_mask_ts)

    i += 1
    depth_pred = cv2.resize(depth_pred, (W, H), interpolation=cv2.INTER_NEAREST)

    if save_figs:
        save_fig_(depth_pred=depth_pred, output_dir=output_dir, after_result=after_result, i=i,
                  depth_raw=depth_raw,name=name,valid_mask=valid_mask_ts)

        i += 1

    valid_mask = valid_mask_ts.numpy()

    depth_raw_ts = depth_raw_ts.to(device)
    print('depth_raw_ts max', depth_raw_ts.max().item())
    print('depth_raw_ts min', depth_raw_ts.min().item())
    valid_mask_ts = valid_mask_ts.to(device)
    print(f'Using {alignment} with {alignment_max_res} resolution')
    # Align with GT using least square
    if disparity is not None:
        alignment='least_square_disparity'
    if "least_square" == alignment:
        depth_pred, scale, shift = align_depth_least_square(
            gt_arr=depth_raw,
            pred_arr=depth_pred,
            valid_mask_arr=valid_mask,
            return_scale_shift=True,
            max_resolution=alignment_max_res,
        )
    elif "least_square_disparity" == alignment:
        # convert GT depth -> GT disparity
        gt_disparity, gt_non_neg_mask = depth2disparity(
            depth=depth_raw, return_mask=True
        )
        # LS alignment in disparity space
        pred_non_neg_mask = depth_pred > 0
        valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

        disparity_pred, scale, shift = align_depth_least_square(
            gt_arr=gt_disparity,
            pred_arr=depth_pred,
            valid_mask_arr=valid_nonnegative_mask,
            return_scale_shift=True,
            max_resolution=alignment_max_res,
        )
        # convert to depth
        disparity_pred = np.clip(
            disparity_pred, a_min=1e-3, a_max=None
        )  # avoid 0 disparity
        depth_pred = disparity2depth(disparity_pred)
    else:
        raise NotImplementedError

    # Clip to dataset min max
    depth_pred = np.clip(
        depth_pred, a_min=min_depth, a_max=max_depth
    )

    # Evaluate (using CUDA if available)
    sample_metric = []
    depth_pred_ts = torch.from_numpy(depth_pred).to(device)

    for met_func in metric_funcs:
        _metric_name = met_func.__name__
        print('_metric_name', _metric_name)
        _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
        print('_metric', _metric)
        sample_metric.append(_metric.__str__())
        metric_tracker.update(_metric_name, _metric)
        group_trackers[gid].update(_metric_name, _metric)

    return i, sample_metric, metric_tracker, group_trackers


def eval_multi(depth_pred_ori, H, W, depth_raw, i, valid_mask_ts, depth_raw_ts, metric_tracker, group_trackers,
               save_figs,
               output_dir, after_result, device, alignment, alignment_max_res, min_depth, max_depth, metric_funcs,
               gid):
    B, H, W = depth_raw.shape
    depth_pred = np.empty((B, H, W), dtype=depth_pred_ori.dtype)
    for b in range(B):
        depth_pred[b] = cv2.resize(depth_pred_ori[b], (W, H), interpolation=cv2.INTER_NEAREST)

    if save_figs:
        for b in range(B):
            depth_pred_single = depth_pred[b]
            depth_raw_single = depth_raw[b]
            save_fig_(depth_pred=depth_pred_single, output_dir=output_dir, after_result=after_result, i=i,
                      depth_raw=depth_raw_single)

            i += 1

    valid_mask = valid_mask_ts.numpy()

    depth_raw_ts = depth_raw_ts.to(device)
    print('depth_raw_ts max', depth_raw_ts.max().item())
    print('depth_raw_ts min', depth_raw_ts.min().item())
    valid_mask_ts = valid_mask_ts.to(device)
    print(f'Using {alignment} with {alignment_max_res} resolution')
    # Align with GT using least square
    if "least_square" == alignment:
        depth_pred, scale, shift = align_depth_least_square(
            gt_arr=depth_raw,
            pred_arr=depth_pred,
            valid_mask_arr=valid_mask,
            return_scale_shift=True,
            max_resolution=alignment_max_res,
        )
    elif "least_square_disparity" == alignment:
        # convert GT depth -> GT disparity
        gt_disparity, gt_non_neg_mask = depth2disparity(
            depth=depth_raw, return_mask=True
        )
        # LS alignment in disparity space
        pred_non_neg_mask = depth_pred > 0
        valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

        disparity_pred, scale, shift = align_depth_least_square(
            gt_arr=gt_disparity,
            pred_arr=depth_pred,
            valid_mask_arr=valid_nonnegative_mask,
            return_scale_shift=True,
            max_resolution=alignment_max_res,
        )
        # convert to depth
        disparity_pred = np.clip(
            disparity_pred, a_min=1e-3, a_max=None
        )  # avoid 0 disparity
        depth_pred = disparity2depth(disparity_pred)
    else:
        raise NotImplementedError

    # Clip to dataset min max
    depth_pred = np.clip(
        depth_pred, a_min=min_depth, a_max=max_depth
    )

    # Evaluate (using CUDA if available)
    sample_metric = []
    depth_pred_ts = torch.from_numpy(depth_pred).to(device)

    for met_func in metric_funcs:
        _metric_name = met_func.__name__
        print('_metric_name', _metric_name)
        print(f'calculate depth_pred_ts {depth_pred_ts.shape} with depth_raw_ts {depth_raw_ts.shape}')
        _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
        print('_metric', _metric)
        sample_metric.append(_metric.__str__())
        metric_tracker.update(_metric_name, _metric)
        group_trackers[gid].update(_metric_name, _metric)

    return i, sample_metric, metric_tracker, group_trackers, depth_pred


def eval_single_aligned(depth_pred, depth_raw, i, valid_mask_ts, depth_raw_ts, save_figs,
                        output_dir, after_result, device, min_depth, max_depth, metric_funcs, name=None):
    """Evaluate one image using an already-aligned prediction.

    This is used by multiview evaluation to diagnose per-image failures under
    the same global scale/shift estimated for the whole multiview sample.
    """
    depth_pred = np.clip(depth_pred, a_min=min_depth, a_max=max_depth)
    valid_mask = valid_mask_ts.numpy()

    if save_figs:
        save_fig_(depth_pred=depth_pred, output_dir=output_dir, after_result=after_result, i=i,
                  depth_raw=depth_raw, name=name, valid_mask=valid_mask)
        i += 1

    depth_pred_ts = torch.from_numpy(depth_pred).to(device)
    depth_raw_ts = depth_raw_ts.to(device)
    valid_mask_ts = valid_mask_ts.to(device)

    sample_metric = []
    for met_func in metric_funcs:
        _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
        sample_metric.append(_metric.__str__())

    return i, sample_metric
