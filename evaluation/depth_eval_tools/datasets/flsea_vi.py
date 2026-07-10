import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class UnderwaterDepthDataset(Dataset):
    def __init__(self, gt_root, debug=False, debug2=False, no_water=False,
                 undistort=False,
                 debug3=False):
        self.data_list = {}
        self.min_depth = 1e-3
        self.max_depth = 30
        self.debug = debug
        self.debug2 = debug2
        self.debug3 = debug3
        self.disp_name = 'FLSea_VI'
        self.filename_ls_path = gt_root

        ########### 畸变系数 ##########

        ## 参数1 calibration
        # TODO: 暂时不管

        self.no_water = no_water
        for scene in os.listdir(gt_root):
            scene_root = os.path.join(gt_root, scene)
            if not os.path.isdir(scene_root):
                print('no scene in {}'.format(scene))
                continue
            for obj in os.listdir(scene_root):
                if obj != 'u_canyon' and obj != 'coral_table_loop':
                    continue
                self.data_list[f'{obj}'] = []
                obj_root = os.path.join(scene_root, obj)
                if not self.no_water:
                    imgs_root = os.path.join(obj_root, "imgs")

                else:
                    imgs_root = os.path.join(obj_root, "seaErra")
                depth_root = os.path.join(obj_root, "depth")
                if not os.path.isdir(imgs_root) or not os.path.isdir(depth_root):
                    print('no file in {}'.format(obj))
                    continue
                for root, _, files in os.walk(imgs_root):
                    for f in files:
                        img_path = os.path.join(root, f)
                        # print(img_path)
                        rel = os.path.relpath(img_path, imgs_root)
                        rel_dir = os.path.dirname(rel)
                        filename = os.path.basename(rel)
                        name, ext = os.path.splitext(filename)
                        name = name.replace('_SeaErra', '')

                        new_filename = f"{name}_SeaErra_abs_depth.tif"

                        # 拼接完整 depth_path
                        depth_path = os.path.join(depth_root, rel_dir, new_filename)
                        # print('depth_path',depth_path)
                        if os.path.exists(depth_path):
                            self.data_list[f'{obj}'].append((scene, obj, img_path, depth_path))

        self.all_to_test = 0
        self.samples = []
        keys = list(self.data_list.keys())
        for obj in keys:
            items = sorted(self.data_list[f'{obj}'])
            if self.debug and not self.debug2:
                k = 10
                items = self.even_sample(items, k)
            elif (not self.debug) and self.debug2:
                k = 100
                items = self.even_sample(items, k)
            else:
                items = items

            nums_to_detect = len(items)

            self.all_to_test += nums_to_detect
            self.samples.extend(items)

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

    def even_indices(self, L: int, k: int):
        """
        返回长度为 L 的序列里，尽量均匀分布的 k 个下标（覆盖首尾）。
        不会越界；k >= L 时返回所有下标。
        """
        if k >= L:
            return list(range(L))
        if k <= 0:
            return []
        if k == 1:
            return [L // 2]
        # 覆盖首尾：i in [0, k-1] 映射到 [0, L-1]
        return [round(i * (L - 1) / (k - 1)) for i in range(k)]

    def even_sample(self, seq, k):
        idx = self.even_indices(len(seq), k)
        return [seq[i] for i in idx]

    def __getitem__(self, idx):
        scene, obj, img_path, depth_path = self.samples[idx]

        # img = Image.open(img_path).convert("RGB")
        # img = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
        #
        depth = Image.open(depth_path)
        depth = np.array(depth, dtype=np.float32)
        # depth = self.threshold_depth_map(depth, min_percentile=2, max_percentile=98)
        depth_ts = torch.from_numpy(depth)
        valid_mask = self._get_valid_mask(depth_ts)

        # 16316027534785523.tiff
        # 16316027536737964_SeaErra_abs_depth.tif

        # return {"image": img, "depth": depth, "image_path": img_path, "depth_path": depth_path}
        return {
            "scene": scene,
            'obj': obj,
            "depth": depth,
            "image_path": img_path,
            "depth_path": depth_path,
            'valid_mask': valid_mask
        }

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
