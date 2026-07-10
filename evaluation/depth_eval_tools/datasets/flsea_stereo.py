import os

import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat


# import tifffile as tiff


class UnderwaterDepthDataset(Dataset):
    def __init__(self, gt_root):
        self.data_list = {}
        self.min_depth = 1e-3
        self.max_depth = 30
        self.disp_name = 'flsea_stereo'
        self.filename_ls_path = gt_root
        self.data_list = []
        for scene in os.listdir(gt_root):
            scene_root = os.path.join(gt_root, scene)
            if not os.path.isdir(scene_root):
                print('no scene in {}'.format(scene))
                continue
            for obj in os.listdir(scene_root):
                obj_root = os.path.join(scene_root, obj)
                imgs_root = os.path.join(obj_root, "imgs")
                depth_root = os.path.join(obj_root, "depth")
                if not os.path.isdir(imgs_root) or not os.path.isdir(depth_root):
                    print('no file in {}'.format(obj))
                    continue
                right_view_root = os.path.join(imgs_root, "RGT")
                right_depth_root = os.path.join(depth_root, "RGT")
                n = 0
                left_view_root = os.path.join(imgs_root, "LFT")
                left_depth_root = os.path.join(depth_root, "LFT")
                for root, _, files in os.walk(right_view_root):
                    for f in files:
                        right_view = os.path.join(root, f)
                        # print('img_path', img_path)
                        filename = os.path.basename(right_view)
                        # print(filename)
                        if filename == 'RGT_01_010578.tif':
                            continue
                        depth_file_name = filename.replace('.tif', '_abs_depth.tif')
                        right_depth = os.path.join(right_depth_root, depth_file_name.replace('LFT', 'RGT'))
                        left_view = os.path.join(left_view_root, filename.replace('RGT', 'LFT'))
                        left_depth = os.path.join(left_depth_root, depth_file_name.replace('RGT', 'LFT'))
                        if os.path.exists(left_depth):
                            self.data_list.append(
                                (scene, obj, right_view, left_view, right_depth, left_depth))
                            n += 1
                print(f'{obj} find {n} image pairs')
        self.all_to_test = len(self.data_list)
        print(f'get {self.all_to_test} images')

    def __len__(self):
        return self.all_to_test

    def __getitem__(self, idx):

        scene, obj, right_view, left_view, right_depth, left_depth = self.data_list[idx]
        depth_left = cv2.imread(left_depth, cv2.IMREAD_UNCHANGED)
        depth_right = cv2.imread(right_depth, cv2.IMREAD_UNCHANGED)
        depth_left = np.nan_to_num(depth_left, nan=-1)
        depth_right = np.nan_to_num(depth_right, nan=-1)
        depth_ts_left = torch.from_numpy(depth_left)
        depth_ts_right = torch.from_numpy(depth_right)
        valid_mask_left = self._get_valid_mask(depth_ts_left)
        valid_mask_right = self._get_valid_mask(depth_ts_right)
        return {
            "scene": scene,
            'obj': obj,
            "depth_left": depth_left,
            "depth_right": depth_right,
            "left_img": left_view,
            'right_img': right_view,
            "right_depth_path": right_depth,
            "left_depth_path": left_depth,
            'valid_mask_left': valid_mask_left,
            'valid_mask_right': valid_mask_right,
        }
    def _get_valid_mask(self, depth: torch.Tensor):
        valid_mask = torch.logical_and(
            (depth > self.min_depth), (depth < self.max_depth)
        ).bool()
        return valid_mask

    def create_dataloader(self, batch_size=1, num_workers=0):
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return dataloader
