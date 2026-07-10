import os

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader


class UnderwaterDepthDataset(Dataset):
    def __init__(self, gt_root, debug=False, debug2=False, no_water=False,
                 undistort=False,
                 debug3=False,
                 ):
        self.data_list = {}
        self.min_depth = 1e-3
        self.max_depth = 30
        self.disp_name = 'SQUID'
        self.filename_ls_path = gt_root

        self.data_list = []
        for scene in os.listdir(gt_root):
            scene_root = os.path.join(gt_root, scene)
            if not os.path.isdir(scene_root):
                print('no scene in {}'.format(scene))
                continue
            for obj in os.listdir(scene_root):

                obj_root = os.path.join(scene_root, obj)
                # print('obj_root',obj_root)
                # i+=1
                # print('i',i)
                for dirpath, dirnames, filenames in os.walk(obj_root):
                    # print(filenames)
                    left_img = next((os.path.join(dirpath, f) for f in filenames
                                     if f.startswith('LFT_') and f.endswith('resizedUndistort.tif')), None)
                    right_img = next((os.path.join(dirpath, f) for f in filenames
                                      if f.startswith('RGT_') and f.endswith('resizedUndistort.tif')), None)
                depth_path = os.path.join(obj_root, 'xyzPoints.mat')
                # depth_path = os.path.join(obj_root, 'distanceFromCamera.mat')
                # print('left_img',left_img)
                # print('right_img',right_img)
                # print('depth_path',depth_path)

                if os.path.exists(depth_path):
                    self.data_list.append((scene, obj, left_img, right_img, depth_path))
                else:
                    print(f'{depth_path} not exist')

        self.all_to_test = len(self.data_list)
        print(f'get {self.all_to_test} images')

    def __len__(self):
        return self.all_to_test

    def threshold_depth_map(self,
                            depth_map: np.ndarray,
                            max_percentile: float = 99,
                            min_percentile: float = 1,
                            max_depth: float = -1,
                            ) -> np.ndarray:
        """
        Thresholds a depth map using percentile-based limits and optional maximum depth clamping.

        Steps:
          1. If `max_depth > 0`, clamp all values above `max_depth` to zero.
          2. Compute `max_percentile` and `min_percentile` thresholds using nanpercentile.
          3. Zero out values above/below these thresholds, if thresholds are > 0.

        Args:
            depth_map (np.ndarray):
                Input depth map (H, W).
            max_percentile (float):
                Upper percentile (0-100). Values above this will be set to zero.
            min_percentile (float):
                Lower percentile (0-100). Values below this will be set to zero.
            max_depth (float):
                Absolute maximum depth. If > 0, any depth above this is set to zero.
                If <= 0, no maximum-depth clamp is applied.

        Returns:
            np.ndarray:
                Depth map (H, W) after thresholding. Some or all values may be zero.
                Returns None if depth_map is None.
        """
        if depth_map is None:
            return None

        depth_map = depth_map.astype(float, copy=True)

        # Optional clamp by max_depth
        if max_depth > 0:
            depth_map[depth_map > max_depth] = 0.0

        # Percentile-based thresholds
        depth_max_thres = (
            np.nanpercentile(depth_map, max_percentile) if max_percentile > 0 else None
        )
        depth_min_thres = (
            np.nanpercentile(depth_map, min_percentile) if min_percentile > 0 else None
        )

        # Apply the thresholds if they are > 0
        if depth_max_thres is not None and depth_max_thres > 0:
            depth_map[depth_map > depth_max_thres] = 0.0
        if depth_min_thres is not None and depth_min_thres > 0:
            depth_map[depth_map < depth_min_thres] = 0.0

        return depth_map

    def __getitem__(self, idx):
        scene, obj, left_img, right_img, depth_path = self.data_list[idx]
        print('left_img', left_img)
        print('right_img', right_img)
        # img = Image.open(img_path).convert("RGB")
        # img = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
        #
        # depth = Image.open(depth_path)
        dist_data = loadmat(depth_path)
        print('dist_data', dist_data.keys())
        pts3d_left = dist_data['xyzPointsLeft']
        pts3d_right = dist_data['xyzPointsRight']
        pts3d_left = np.array(pts3d_left, dtype=np.float32)

        pts3d_right = np.array(pts3d_right, dtype=np.float32)

        depth_left = pts3d_left[:, :, 2]
        depth_right = pts3d_right[:, :, 2]
        # depth_left=dist_data['dist_map_l']
        # depth_right=dist_data['dist_map_r']
        print("depth_left count > 30:", int(np.sum(depth_left > 30)))
        print("depth_left count > 1000:", int(np.sum(depth_left > 1000)))
        print("depth_right count > 30:", int(np.sum(depth_right > 30)))
        print("depth_right count > 1000:", int(np.sum(depth_right > 1000)))
        # depth_left = self.threshold_depth_map(depth_left, min_percentile=2, max_percentile=98)
        # depth_right = self.threshold_depth_map(depth_right, min_percentile=2, max_percentile=98)
        depth_left = np.nan_to_num(depth_left, nan=0.)
        depth_right = np.nan_to_num(depth_right, nan=0.)
        # 打印非NaN的数量
        # num_valid_left = np.sum(~np.isnan(depth_left))
        # num_valid_right = np.sum(~np.isnan(depth_right))
        #
        # print("左图非NaN像素数:", num_valid_left)
        # print("右图非NaN像素数:", num_valid_right)
        depth_ts_left = torch.from_numpy(depth_left)
        depth_ts_right = torch.from_numpy(depth_right)
        valid_mask_left = self._get_valid_mask(depth_ts_left)
        valid_mask_right = self._get_valid_mask(depth_ts_right)

        # 16316027534785523.tiff
        # 16316027536737964_SeaErra_abs_depth.tif

        # return {"image": img, "depth": depth, "image_path": img_path, "depth_path": depth_path}
        return {
            "scene": scene,
            'obj': obj,
            "depth_left": depth_left,
            "depth_right": depth_right,
            "left_img": left_img,
            'right_img': right_img,
            "depth_path": depth_path,
            'valid_mask_left': valid_mask_left,
            'valid_mask_right': valid_mask_right,
        }

    #
    # def _get_valid_mask(self, depth: torch.Tensor):
    #     valid_mask = (depth > self.min_depth).bool()
    #     return valid_mask
    def _get_valid_mask(self, depth: torch.Tensor):
        valid_mask = torch.logical_and(
            (depth > self.min_depth), (depth < self.max_depth)
        ).bool()
        return valid_mask

    def create_dataloader(self, batch_size=1, num_workers=0):
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return dataloader
