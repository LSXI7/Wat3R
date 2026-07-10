from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree as KDTree


def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_tree = KDTree(gt_points)
    distances, indices = gt_tree.query(rec_points, workers=24)
    acc = np.mean(distances)
    acc_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals[indices] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)
        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

    return acc, acc_median


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    rec_tree = KDTree(rec_points)
    distances, indices = rec_tree.query(gt_points, workers=24)
    comp = np.mean(distances)
    comp_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals * rec_normals[indices], axis=-1)
        normal_dot = np.abs(normal_dot)
        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)

    return comp, comp_median

